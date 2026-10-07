"""The ExoBoot four-point spline: plantarflexion torque as a PCHIP spline over the full gait cycle.

Port of ``GenericSplineController`` + ``FourPointSplineController`` (ExoBoot controllers.py:328-437).
Knots are absolute positions on a 0-1 gait-cycle axis (heel strike to heel strike), which is *not* how
MyoAssist's own ``FourParamSplineController`` is parameterized (durations either side of a peak, on a
percent-of-stance axis). The two are not interchangeable; this one exists so that the parameters tuned on
the physical boot mean the same thing here.

Deliberately not ported:

* ``fade_splines``, a 5 s wall-clock cross-fade for retuning parameters live on hardware. Parameters are
  fixed for an RL run.
* The ``phase > spline_x[-1]`` branch of ``command()``, which evaluates the spline at the whole knot array
  and so returns an array, not a torque. It is unreachable because phase is clipped to 1.
* The reel-in, reel-out and swing ("stalk") controllers that surround the spline on hardware. They manage
  Bowden-cable slack and apply no assist; what survives of them here is that torque is zero outside stance.
"""

from __future__ import annotations

import math

from scipy.interpolate import PchipInterpolator

from myoassist_utils.exo_ctrl.base import HeelStrikeDetector
from myoassist_utils.exo_ctrl.phase import StrideAverageGaitPhaseEstimator


class FourPointSpline:
    """The torque profile alone: a pure function of gait phase, with no state.

    With the ExoBoot's tuned defaults (rise 0.278, peak 0.543, fall 0.641, 25 N*m, bias 3 N*m) torque sits at
    the bias, rises from ``rise_fraction`` to ``peak_torque`` at ``peak_fraction``, and falls back to the bias
    by ``fall_fraction``. ``peak_hold_time`` > 0 holds the peak for that fraction of the cycle.
    """

    def __init__(
        self,
        *,
        rise_fraction: float,
        peak_fraction: float,
        fall_fraction: float,
        peak_torque: float,
        bias_torque: float,
        peak_hold_time: float = 0.0,
    ):
        params = dict(
            rise_fraction=rise_fraction,
            peak_fraction=peak_fraction,
            fall_fraction=fall_fraction,
            peak_torque=peak_torque,
            bias_torque=bias_torque,
            peak_hold_time=peak_hold_time,
        )
        bad = [name for name, value in params.items() if not math.isfinite(value)]
        if bad:
            raise ValueError(f"four-point spline parameters must be finite: {bad}")
        # The ankle exos can only plantarflex (ctrlrange [-1, 0]), so a negative torque would be clipped to zero
        # and the profile would silently not be the one configured.
        if peak_torque < 0 or bias_torque < 0:
            raise ValueError(
                f"torques must be >= 0 (plantarflexion-positive, and the actuator cannot dorsiflex); "
                f"got peak={peak_torque}, bias={bias_torque}"
            )
        if peak_hold_time < 0:
            raise ValueError(f"peak_hold_time must be >= 0, got {peak_hold_time}")

        # Same knots as ExoBoot _get_spline_x / _get_spline_y (controllers.py:427-437).
        if peak_hold_time > 0:
            x = [
                0.0,
                rise_fraction,
                peak_fraction,
                peak_fraction + peak_hold_time,
                fall_fraction,
                1.0,
            ]
            y = [
                bias_torque,
                bias_torque,
                peak_torque,
                peak_torque,
                bias_torque,
                bias_torque,
            ]
        else:
            x = [0.0, rise_fraction, peak_fraction, fall_fraction, 1.0]
            y = [bias_torque, bias_torque, peak_torque, bias_torque, bias_torque]
        # PCHIP needs strictly increasing knots. Checking here fails at env construction rather than raising
        # from scipy, and names the parameters instead of the knot array.
        if any(b <= a for a, b in zip(x, x[1:])):
            hold = " < peak_fraction + peak_hold_time" if peak_hold_time > 0 else ""
            raise ValueError(f"need 0 < rise_fraction < peak_fraction{hold} < fall_fraction < 1 strictly; knots were {x}")

        self.params = params
        self.knots_x = tuple(x)
        self.knots_y = tuple(y)
        # extrapolate=False matches the hardware. It returns NaN outside [0, 1] -- including at 1 + 1e-12 --
        # so torque() clamps phase first. One NaN reaching the action poisons a whole PPO batch.
        self._spline = PchipInterpolator(x, y, extrapolate=False)

    def torque(self, phase: float | None) -> float:
        """Plantarflexion torque in N*m. No phase (gait not steady) means no torque, as on hardware."""
        if phase is None:
            return 0.0
        return float(self._spline(min(1.0, max(0.0, phase))))


# The boot's control states (NeuMoveExoBoot constants.ControlState) this port reproduces. The boot's reel-out (1), which
# lets its cable out after toe-off, applies no assist either; with no cable here it is reported as swing.
SWING_STATE, REEL_IN_STATE, STANCE_STATE = 2, 3, 4


class ExoBootFourPointSplineController:
    """One leg's controller: heel strikes -> gait phase -> spline torque, in the ExoBoot's order.

    Each tick detects a heel strike, then estimates phase, then evaluates the spline, the same sequence as
    ``GaitStateEstimator.detect`` followed by ``GenericSplineController.command`` on hardware.

    The spline only drives the boot during stance. On hardware it is the stance controller of
    ``StanceSwingReeloutReelinStateMachine``, which hands over to reel-out and then swing (no assist) as soon
    as gait phase passes ``TOE_OFF_FRACTION`` (``GaitPhaseBasedToeOffDetector``). So the torque the boot
    delivers is the spline truncated at ``toe_off_fraction`` -- and with the tuned knots it is cut mid-fall,
    since ``fall_fraction`` (0.641) lies past toe-off (0.60): torque drops from 9.5 N*m to zero there (11.3 with the
    boot's 3 N*m bias), and the bias is never applied in swing. Evaluating the spline over the whole cycle instead delivers 23% more
    impulse than the boot does. Setting ``toe_off_fraction`` to 1 gives that ungated profile, if wanted.

    Stance does not start at the heel strike either. The state machine first reels in (``SmoothReelInController``:
    voltage control until the cable's slack falls below ``REEL_IN_SLACK_CUTOFF``, or for 0.2 s at most, a timeout
    hard-coded there -- the ``REEL_IN_TIMEOUT`` it is passed is not used), and only then runs the spline. ``reel_in_time`` stands
    in for that, since there is no cable here: no torque for that long after each detected strike. It is a fixed
    time, not a phase -- in the validation session's log, reel-in took 150 / 164 ms (left / right) at 1.25 m/s, and across
    speed changes (75-127 steps/min) it did not follow step frequency: +0.2 +/- 0.4 and +0.7 +/- 0.5 ms per
    step/min (left / right, 95% CI), where a fixed fraction of the gait cycle would give about -1.5. The boot
    commands no spline torque during reel-in and delivers ~0.8 N*m of cable tension; applying the bias there
    instead adds 7-8% to its impulse.

    ``diagnostics()`` reports the state the boot would log as ``control_state``: swing (2) until a gait phase exists
    -- the boot only enters reel-in at a heel strike once it has one -- then reel-in (3), stance (4), and swing (2)
    again from toe-off to the next strike.

    The env steps it at a fixed rate inside the physics loop (``device.FixedRateLegExos``), as the boot's own loop
    does, and holds its torque between ticks. Timing is all in sim time and gait phase, so the rate only sets how
    finely heel strikes and gates are resolved: one tick, 6.7 ms at 150 Hz.
    """

    def __init__(
        self,
        *,
        spline: FourPointSpline,
        heel_strike_detector: HeelStrikeDetector,
        phase_estimator: StrideAverageGaitPhaseEstimator,
        toe_off_fraction: float = 0.60,
        reel_in_time: float = 0.0,
    ):
        if not (math.isfinite(reel_in_time) and reel_in_time >= 0):
            raise ValueError(f"reel_in_time must be finite and >= 0, got {reel_in_time}")
        if not (math.isfinite(toe_off_fraction) and 0 < toe_off_fraction <= 1):
            raise ValueError(f"toe_off_fraction must be in (0, 1], got {toe_off_fraction}")
        if not toe_off_fraction > spline.params["rise_fraction"]:
            raise ValueError(
                f"toe_off_fraction ({toe_off_fraction}) must be after rise_fraction ({spline.params['rise_fraction']}), "
                "or stance ends before any assist"
            )
        self.spline = spline
        self.heel_strike_detector = heel_strike_detector
        self.phase_estimator = phase_estimator
        self.toe_off_fraction = toe_off_fraction
        self.reel_in_time = reel_in_time
        self.reset()

    def reset(self) -> None:
        self.heel_strike_detector.reset()
        self.phase_estimator.reset()
        self._did_heel_strike = False
        self._phase: float | None = None
        self._in_stance = False
        self._control_state = SWING_STATE
        self._torque = 0.0

    def step(self, t: float, signal: float) -> float:
        detector = self.heel_strike_detector
        self._did_heel_strike = detector.detect(t, signal)
        if self._did_heel_strike and detector.strike_time < t:
            # A strike confirmed after the fact (VgrfHeelStrikeDetector.min_contact_time) starts its stride when it
            # happened, so the phase now is the time since then.
            self.phase_estimator.estimate(detector.strike_time, True)
            self._phase = self.phase_estimator.estimate(t, False)
        else:
            self._phase = self.phase_estimator.estimate(t, self._did_heel_strike)
        phase = self._phase
        if phase is None:
            self._in_stance = False
            self._control_state = SWING_STATE
        else:
            # A phase is only returned once a mean stride duration exists. Reel-in is a time, so it becomes a phase
            # through the same stride average the phase itself is divided by. Toe-off is strictly greater, as
            # GaitPhaseBasedToeOffDetector fires when phase > TOE_OFF_FRACTION.
            stride = self.phase_estimator.mean_stride_duration
            self._in_stance = self.reel_in_time / stride <= phase <= self.toe_off_fraction
            if self._in_stance:
                self._control_state = STANCE_STATE
            elif phase <= self.toe_off_fraction:
                self._control_state = REEL_IN_STATE
            else:
                self._control_state = SWING_STATE
        self._torque = self.spline.torque(phase) if self._in_stance else 0.0
        return self._torque

    def diagnostics(self) -> dict[str, float]:
        # Phase is reported as -1 when invalid rather than NaN: these values can be summed into training logs,
        # and -1 is outside [0, 1] so it cannot be mistaken for a real phase. The stride estimate (s) is the mean of
        # the last strides the phase is divided by, -1 before there is one.
        return {
            "torque_nm": self._torque,
            "phase": -1.0 if self._phase is None else self._phase,
            "phase_valid": 0.0 if self._phase is None else 1.0,
            "in_stance": 1.0 if self._in_stance else 0.0,
            "heel_strike": 1.0 if self._did_heel_strike else 0.0,
            "control_state": float(self._control_state),
            "stride_estimate": -1.0
            if self.phase_estimator.mean_stride_duration is None
            else self.phase_estimator.mean_stride_duration,
        }
