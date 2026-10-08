"""What the ExoBoot's controllers share around their stance controller: its state machine, and its ankle reading.

* ``BootStateMachine`` is ``StanceSwingReeloutReelinStateMachine.step`` on sim time: reel-in on a heel strike, stance
  once reel-in completes, reel-out on toe-off, swing once reel-out completes. Only stance applies torque. The boot ends
  reel-in and reel-out on cable slack or a timeout; there is no cable here, so they end after fixed durations, measured
  on the boot's logs.
* ``AnkleEncoder`` is the ankle angle and velocity as ``Exo.read_data`` (exoboot.py:313-379) reads them: degrees,
  plantarflexion-positive, encoder clicks (360/2^14 deg each) plus a fixed per-side offset; the velocity is the angle's
  difference over the pack's timestamps through a 2nd-order 10 Hz Butterworth designed for the loop rate, 0 and
  unfiltered on the first read.

Plain Python floats throughout, since they run inside the physics loop.
"""

from __future__ import annotations

import math

import mujoco

from myoassist_utils.exo_ctrl.boot_filters import Butterworth, DelayTimer

# constants.ControlState
REEL_OUT, SWING, REEL_IN, STANCE = 1, 2, 3, 4

ANKLE_JOINT = "ankle_angle_{side}"
ENCODER_CLICK_DEG = 360.0 / 2**14  # constants.ENC_CLICKS_TO_DEG
ANKLE_VELOCITY_FILTER = dict(order=2, cutoff_hz=10.0)  # exoboot.py:134, designed at the loop rate


class BootStateMachine:
    """``StanceSwingReeloutReelinStateMachine.step`` for a stance controller, on sim time.

    Reel-in and reel-out end when their ``DelayTimer`` runs out (strictly after the duration), as the boot's do when
    the slack test does not end them first. The boot starts reel-in on the tick its heel-strike detector fires, which is
    the strike. A detector that confirms a strike after the fact (``VgrfHeelStrikeDetector.min_contact_time``) passes
    ``strike_time``, when the strike happened, and reel-in is timed from then.
    """

    def __init__(self, *, reel_in_time: float, reel_out_time: float):
        for name, value in (("reel_in_time", reel_in_time), ("reel_out_time", reel_out_time)):
            if not value >= 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        self._reel_in = DelayTimer(reel_in_time)
        self._reel_out = DelayTimer(reel_out_time)
        self.reset()

    def reset(self) -> None:
        self.state: int | None = None
        self._just_starting = True
        self._toe_off_switch = False
        self._reel_in.reset()
        self._reel_out.reset()

    def step(
        self,
        t: float,
        *,
        did_heel_strike: bool,
        did_toe_off: bool,
        gait_phase: float | None,
        swing_only: bool,
        strike_time: float | None = None,
    ) -> int:
        if self.state == STANCE and (did_toe_off or gait_phase is None):
            self._toe_off_switch = True
        if self._just_starting:
            self._just_starting = False
            self.state = REEL_OUT
            self._reel_out.start(t)
        elif swing_only:
            self.state = SWING
        elif self.state == SWING and did_heel_strike and gait_phase is not None:
            self.state = REEL_IN
            self._reel_in.start(t if strike_time is None else min(t, strike_time))
        elif self.state == REEL_IN and self._reel_in.check(t):
            self._reel_in.reset()
            self.state = STANCE
        elif self._toe_off_switch:
            self._toe_off_switch = False
            self.state = REEL_OUT
            self._reel_out.start(t)
        elif self.state == REEL_OUT and self._reel_out.check(t):
            self.state = SWING
        return self.state


class AnkleEncoder:
    """One leg's ankle angle and velocity in the boot's units and sign, read once per controller tick.

    ``standing_angle_deg`` is what the boot's ankle angle reads at the standing keyframe: it fixes the offset the boot
    adds to its encoder angle. ``quantize`` rounds the angle to whole encoder clicks first, as the encoder does.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        side: str,
        *,
        standing_angle_deg: float,
        sample_rate_hz: float = 175.0,
        quantize: bool = True,
    ):
        self.side = side
        self.quantize = quantize
        joint = model.joint(ANKLE_JOINT.format(side=side))
        self._qpos = int(model.jnt_qposadr[joint.id])
        standing_q = float(model.key_qpos[0][self._qpos]) if model.nkey else 0.0
        # Boot angle = -(model angle in deg) + offset, the offset putting the standing keyframe at the standing angle.
        self.offset_deg = standing_angle_deg + math.degrees(standing_q)
        self._velocity_filter = Butterworth(fs_hz=sample_rate_hz, **ANKLE_VELOCITY_FILTER)
        self.reset()

    def reset(self) -> None:
        self._velocity_filter.reset()
        self._last_angle: float | None = None
        self._last_time: float | None = None
        self._velocity = 0.0

    def read(self, sim) -> tuple[float, float]:
        """(angle in deg, velocity in deg/s), plantarflexion-positive."""
        data = sim.data
        encoder = -math.degrees(float(data.qpos[self._qpos]))
        if self.quantize:
            encoder = round(encoder / ENCODER_CLICK_DEG) * ENCODER_CLICK_DEG
        angle = encoder + self.offset_deg
        t = float(data.time)
        if self._last_angle is not None and t > self._last_time:
            self._velocity = self._velocity_filter.filter((angle - self._last_angle) / (t - self._last_time))
        self._last_angle, self._last_time = angle, t
        return angle, self._velocity
