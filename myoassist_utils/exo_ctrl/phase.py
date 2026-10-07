"""Gait phase for the scripted exo controllers: heel-strike detection and a stride-average phase estimate."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from myoassist_utils.exo_ctrl.boot_filters import Butterworth, DelayTimer


class GyroHeelStrikeDetector:
    """Port of the ExoBoot's ``GyroHeelStrikeDetector`` (gait_state_estimators.py:350-369), with its WALKING-task filter.

    A heel strike is reported a fixed delay after the swing-phase peak of the shank's sagittal angular velocity:

    * ``gyro_z`` is low-passed by the boot's Butterworth (``HS_GYRO_FILTER_N`` 2, ``HS_GYRO_FILTER_WN`` 3 Hz, designed
      for ``TARGET_FREQ`` 175 Hz; control_muxer.py:27-32);
    * a filtered sample above ``threshold`` (``HS_GYRO_THRESHOLD``, deg/s) that exceeds both of its neighbours is a
      peak. That can only be known on the next sample, so the test is one sample late;
    * each peak (re)starts a ``DelayTimer`` of ``delay`` s (``HS_GYRO_DELAY``), and the strike is the first sample
      strictly after it runs out. A second peak before then restarts it.

    ``signal`` is ``gyro_z`` in deg/s as the boot's ``Exo.read_data`` converts it, with z pointing laterally outward on
    both legs, so forward swing reads positive on both. The filter assumes ``sample_rate_hz``: step once per tick.
    """

    def __init__(
        self,
        *,
        threshold: float = 100.0,
        filter_order: int = 2,
        filter_cutoff_hz: float = 3.0,
        sample_rate_hz: float = 175.0,
        delay: float = 0.05,
    ):
        self.threshold = threshold
        self._filter = Butterworth(order=filter_order, cutoff_hz=filter_cutoff_hz, fs_hz=sample_rate_hz)
        self._timer = DelayTimer(delay)
        self.reset()

    def reset(self) -> None:
        self._filter.reset()
        self._timer.reset()
        self._history = [
            0.0,
            0.0,
            0.0,
        ]  # newest first: the boot's deque([0, 0, 0], maxlen=3) with appendleft
        self.strike_time = -math.inf

    def detect(self, t: float, signal: float) -> bool:
        history = self._history
        history[2] = history[1]
        history[1] = history[0]
        history[0] = self._filter.filter(signal)
        if history[1] > self.threshold and history[1] > history[0] and history[1] > history[2]:
            self._timer.start(t)
        if self._timer.check(t):
            self._timer.reset()
            self.strike_time = t
            return True
        return False


class VgrfHeelStrikeDetector:
    """Heel strike from one leg's vertical GRF, as a rising edge through a hysteresis band.

    Stands in for the ExoBoot's ``GyroHeelStrikeDetector``: the MyoAssist models carry no gyro sensor, and
    GRF is exact in simulation.

    Thresholds are raw newtons from ``MyoAssistLegBase._get_foot_force(side)`` (foot + toes touch sensors),
    deliberately not normalized by bodyweight. The touch sensors do not capture full GRF -- on
    ``DephyExoBoot_L1`` all four sum to about 0.49 BW at quiet standing -- so a bodyweight fraction would
    look calibrated without being so.

    Hysteresis alone does not reject a brief unload and re-contact -- the bounce after heel strike, a foot
    rolling, or the chattery contacts of a policy early in training -- which would register a second strike
    and split one stride in two. A split stride is shorter than ``min_stride_duration``, so it knocks phase
    to ``None`` and the exo cuts out for two strides. So a contact is only released once the foot has stayed
    below ``grf_off`` for ``min_unload_time`` seconds. Guarding only the first moments after the strike is
    not enough: an unload later in stance splits the stride just the same. Strikes are not delayed by this,
    and a real swing (~0.4 s) always outlasts it.

    A brief touch in swing is not a heel strike either. A foot that scuffs the ground mid-swing loads it for a few
    milliseconds -- 2-30 ms, at up to ~700 N, on the MyoAssist tutorial policy's right foot -- which crosses any
    ``grf_on`` that real strikes also cross, and splits the stride just as a bounce would. So a contact is only
    reported once it has lasted ``min_contact_time`` and the foot is loaded at that tick, and the strike is dated
    back to when the contact began (``strike_time``): the stride, the phase and the reel-in all start there, so the
    debounce delays nothing the controller does. A contact released before then was a touch, not a strike.

    A foot already loaded when the detector starts is not a heel strike. The first tick after ``reset()``
    only adopts the current contact state; a strike needs an observed unload (below ``grf_off``) first.
    Without this, an episode that starts mid-stance registers a strike on its first tick, and the interval
    from there to the first real strike -- shorter than a stride, but often inside the stride bounds -- is
    accepted and biases the first valid phase. The gyro detector on hardware cannot do this, since it fires
    on the swing-phase angular-velocity peak and a standing foot has none.
    """

    def __init__(self, *, grf_on: float, grf_off: float, min_unload_time: float, min_contact_time: float = 0.0):
        if not grf_on > grf_off >= 0:
            raise ValueError(f"need grf_on > grf_off >= 0 for a hysteresis band, got on={grf_on}, off={grf_off}")
        if min_unload_time < 0:
            raise ValueError(f"min_unload_time must be >= 0, got {min_unload_time}")
        if min_contact_time < 0:
            raise ValueError(f"min_contact_time must be >= 0, got {min_contact_time}")
        self.grf_on = grf_on
        self.grf_off = grf_off
        self.min_unload_time = min_unload_time
        self.min_contact_time = min_contact_time
        self.reset()

    def reset(self) -> None:
        self.in_contact = False
        self._t_unload_start: float | None = None
        self._t_contact_start: float | None = None  # the current contact's onset, until it is reported as a strike
        self._primed = False
        self.strike_time = -math.inf

    def detect(self, t: float, signal: float) -> bool:
        if not self._primed:
            # Anything above the release threshold counts as already in contact, so a foot resting inside the
            # hysteresis band must still unload before it can strike.
            self._primed = True
            self.in_contact = signal > self.grf_off
            return False
        if not self.in_contact:
            if signal <= self.grf_on:
                return False
            self.in_contact = True
            self._t_contact_start = t
        if signal >= self.grf_off:
            self._t_unload_start = None  # reloaded before the unload counted: a bounce, not a toe-off
        elif self._t_unload_start is None:
            self._t_unload_start = t
        if self._t_unload_start is not None and t - self._t_unload_start >= self.min_unload_time:
            self.in_contact = False
            self._t_unload_start = None
            self._t_contact_start = None  # released before it was reported: a touch, not a strike
            return False
        if self._t_contact_start is not None and t - self._t_contact_start >= self.min_contact_time and signal >= self.grf_off:
            self.strike_time = self._t_contact_start
            self._t_contact_start = None
            return True
        return False


class StrideAverageGaitPhaseEstimator:
    """Port of the ExoBoot's ``StrideAverageGaitPhaseEstimator`` (gait_state_estimators.py:393-440).

    Phase is the time since the last heel strike over the mean of recent stride durations, clipped to 1.
    It is ``None`` unless gait looks steady: each of the last ``num_strides_required`` strides must lie in
    ``(min_stride_duration, max_stride_duration)``, and the current stride must not have run past
    ``1.2 * max_stride_duration``.

    Two changes from hardware, both needed for the estimator to mean the same thing in simulation:

    * Time is passed in (simulation time) rather than read from ``time.perf_counter()``.
    * The last heel strike starts at ``-inf`` rather than ``0``. On hardware ``0`` serves as "a long time
      ago" because ``perf_counter()`` is large, so the first stride comes out huge and is rejected.
      Simulation time starts at 0, so keeping ``0`` would count the interval from episode start to the
      first heel strike as a stride -- one that can pass the bounds and bias the average.
    """

    def __init__(
        self,
        *,
        num_strides_required: int = 2,
        num_strides_to_average: int = 2,
        min_stride_duration: float = 0.6,
        max_stride_duration: float = 2.0,
    ):
        if num_strides_required < 1:
            raise ValueError(f"num_strides_required must be >= 1, got {num_strides_required}")
        # The hardware raises the same condition with a message that reads the other way round. The check
        # is what matters: averaging over at most the strides that were validated keeps the mean finite and
        # built only from steady strides whenever a phase is returned.
        if not 1 <= num_strides_to_average <= num_strides_required:
            raise ValueError(
                f"num_strides_to_average must be in [1, num_strides_required={num_strides_required}], "
                f"got {num_strides_to_average}"
            )
        if not 0 < min_stride_duration < max_stride_duration:
            raise ValueError(
                f"need 0 < min_stride_duration < max_stride_duration, got {min_stride_duration}, {max_stride_duration}"
            )
        self.num_strides_required = num_strides_required
        self.num_strides_to_average = num_strides_to_average
        self.min_stride_duration = min_stride_duration
        self.max_stride_duration = max_stride_duration
        self.reset()

    def reset(self) -> None:
        self._t_last_heel_strike = -math.inf
        # The hardware seeds with 1000 s; any out-of-bounds value works, and inf is plainly one.
        self._last_stride_durations = deque([math.inf] * self.num_strides_required, maxlen=self.num_strides_required)
        self._averaging_window = deque(maxlen=self.num_strides_to_average)
        self.mean_stride_duration: float | None = None

    def estimate(self, t: float, did_heel_strike: bool) -> float | None:
        if did_heel_strike:
            stride_duration = t - self._t_last_heel_strike
            self._last_stride_durations.append(stride_duration)
            self._averaging_window.append(stride_duration)
            self._t_last_heel_strike = t
            self.mean_stride_duration = float(np.mean(self._averaging_window))

        time_since_heel_strike = t - self._t_last_heel_strike
        is_steady = (
            all(self.min_stride_duration < d < self.max_stride_duration for d in self._last_stride_durations)
            and time_since_heel_strike < 1.2 * self.max_stride_duration
        )
        if not is_steady:
            return None
        return min(1.0, time_since_heel_strike / self.mean_stride_duration)
