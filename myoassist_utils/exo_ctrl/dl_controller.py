"""The ExoBoot's DL task (``Task.WALKJOGDLGAITPHASE``, as in mle_config), as a controller for the in-loop seam.

Per 175 Hz tick, in the boot's order (main_loop.py: read, detect, state machine, command):

1. Each leg's 8 sensor channels go to the network, rounded to 5 decimals as the Pi sends them. The network runs on
   each leg's zero-filled 200-sample buffer (``gait_net.StreamingGaitNet``).
2. Each leg uses the newest reply, computed from its buffer one tick earlier: the Pi sends this tick's sample just
   before it reads the replies, so the reply to it arrives a tick later. The Jetson sends the stance-phase head to
   5 decimals and the stance/swing head rounded to 0 or 1.
3. ``DLGAITSTATEESTIMATOR``, as the boot's logs show it ran: heel strike and toe-off are the rising and falling edges of
   is_stance; gait phase is 0.6 x the stance-phase head, clipped to [0, 1]; the speed head is low-passed by a
   2nd-order Butterworth designed for 0.5 Hz at 200 Hz. (The committed code also runs a gyro estimator that would
   overwrite the events, but on the boot it did not: every logged event is an is_stance edge.)
4. ``StanceSwingReeloutReelinStateMachine`` (``boot_state.BootStateMachine``): reel-in on a heel strike, stance once reel-in completes, reel-out on
   toe-off, swing once reel-out completes. Only stance applies torque: the four-point spline at the gait phase. The
   boot ends reel-in and reel-out on cable slack or a 0.2 s timeout; there is no cable here, so they end after the
   durations measured on the DL validation sessions' logs.
5. Assistance is gated by ``swing_only``: off at start, and turned on and off by the speed activation criterion
   (``DL_SPEEDACTIVATION``: on when the filtered speed crosses up through 0.7 m/s, off when it falls through 0.5).

Per-tick code stays in plain floats except the network, which is one matmul call per layer for both legs.
"""

from __future__ import annotations

import pathlib
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from myoassist_utils.exo_ctrl.boot_filters import Butterworth
from myoassist_utils.exo_ctrl.boot_sensors import BootSensorFrontEnd, load_out_of_plane_filters
from myoassist_utils.exo_ctrl.boot_state import STANCE, SWING, BootStateMachine
from myoassist_utils.exo_ctrl.fourpoint_spline import FourPointSpline
from myoassist_utils.exo_ctrl.gait_net import INPUT_CHANNELS, StreamingGaitNet, load_weights
from myoassist_utils.exo_ctrl.schedule import TickSchedule
from myoassist_utils.exo_ctrl.torque_adapter import PLANTARFLEXION_SIGN, ankle_torque_actuator

# The network was trained on the boot's 175 Hz samples; its window, filters and lags count those samples.
DL_RATE_HZ = 175.0
GAIT_PHASE_SCALE = 0.6
IS_STANCE_THRESHOLD = 0.5
SPEED_FILTER = dict(order=2, cutoff_hz=0.5, fs_hz=200.0)  # gait_state_estimators.py:181, Butterworth(2, 0.5, 'low', 200)
NET_HEADS = ("stance_swing", "stance_phase", "velocity")  # what the boot's control uses; ramp is only logged
ASSIST_ON = ("speed", "always")
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


class SpeedActivation:
    """state_machines.py:164-175: assist on when the filtered speed crosses up through ``on_speed``, off when it crosses
    down through ``off_speed``. Starts from a previous speed of 0, as the boot's ``deque([0, 0])`` does."""

    def __init__(self, *, on_speed: float = 0.7, off_speed: float = 0.5):
        if not off_speed < on_speed:
            raise ValueError(f"need off_speed < on_speed, got {off_speed}, {on_speed}")
        self.on_speed, self.off_speed = on_speed, off_speed
        self.reset()

    def reset(self) -> None:
        self._previous = 0.0

    def update(self, speed: float, swing_only: bool) -> bool:
        previous, self._previous = self._previous, speed
        if previous < self.on_speed and speed >= self.on_speed:
            return False
        if previous > self.off_speed and speed <= self.off_speed:
            return True
        return swing_only


class DLLeg:
    """One leg's estimator, state machine and torque, from that leg's network replies."""

    def __init__(self, *, spline: FourPointSpline, state_machine: BootStateMachine, assist_on: str, speed: SpeedActivation):
        if assist_on not in ASSIST_ON:
            raise ValueError(f"assist_on must be one of {ASSIST_ON}, got {assist_on!r}")
        self.spline = spline
        self.state_machine = state_machine
        self.assist_on = assist_on
        self.speed = speed
        self._speed_filter = Butterworth(**SPEED_FILTER)
        self.reset()

    def reset(self) -> None:
        self.state_machine.reset()
        self.speed.reset()
        self._speed_filter.reset()
        self.swing_only = self.assist_on != "always"
        self._last_is_stance = False
        self.did_heel_strike = self.did_toe_off = False
        self.gait_phase: float | None = None
        self.stance_phase: float | None = None
        self.filtered_speed: float | None = None
        self.reply: tuple[float, float, float] | None = None
        self.sent: Sequence[float] | None = None  # the 8 channels this leg sent on the last tick (set by the device)
        self.state: int | None = None
        self.torque = 0.0

    def step(self, t: float, reply: tuple[float, float, float] | None) -> float:
        """``reply`` is (stance phase, is_stance, speed) as received, or None before the first reply."""
        if reply is not None:  # DLGAITSTATEESTIMATOR.detect returns early, changing nothing, until a reply arrives
            self.reply = reply
            stance_phase, is_stance, speed = reply
            self.stance_phase = min(1.0, max(0.0, stance_phase))
            on = is_stance > IS_STANCE_THRESHOLD
            self.did_heel_strike = on and not self._last_is_stance
            self.did_toe_off = self._last_is_stance and not on
            self._last_is_stance = on
            self.gait_phase = GAIT_PHASE_SCALE * self.stance_phase
            self.filtered_speed = self._speed_filter.filter(speed)
        if self.assist_on == "speed" and self.filtered_speed is not None:
            self.swing_only = self.speed.update(self.filtered_speed, self.swing_only)
        self.state = self.state_machine.step(
            t,
            did_heel_strike=self.did_heel_strike,
            did_toe_off=self.did_toe_off,
            gait_phase=self.gait_phase,
            swing_only=self.swing_only,
        )
        self.torque = self.spline.torque(self.gait_phase) if self.state == STANCE else 0.0
        return self.torque

    def diagnostics(self) -> dict[str, float]:
        """Always finite, for training logs. phase_valid: assisting is possible this tick (assist on, a phase known).

        Also, as the boot logs them: ``control_state`` (``boot_state``'s codes; swing before the first tick), the reply
        as received (``is_stance`` 0/1, ``stance_phase_head`` unclipped, ``speed_head``; -1 before the first reply), and
        what this leg sent on the last tick (``gyro_x``, ``gyro_y``, ``gyro_z``, deg/s; 0 before it)."""
        reply = (-1.0, -1.0, -1.0) if self.reply is None else self.reply
        sent = self.sent if self.sent is not None else (0.0,) * len(INPUT_CHANNELS)
        return {
            "torque_nm": self.torque,
            "phase": -1.0 if self.gait_phase is None else self.gait_phase,
            "phase_valid": float(not self.swing_only and self.gait_phase is not None),
            "in_stance": float(self.state == STANCE),
            "heel_strike": float(self.did_heel_strike),
            "assist_on": float(not self.swing_only),
            "speed": -1.0 if self.filtered_speed is None else self.filtered_speed,
            "control_state": float(SWING if self.state is None else self.state),
            "is_stance": reply[1],
            "stance_phase_head": reply[0],
            "speed_head": reply[2],
            "gyro_x": float(sent[3]),
            "gyro_y": float(sent[4]),
            "gyro_z": float(sent[5]),
        }


@dataclass
class DLLegDrive:
    """A leg as the env sees it: its actuator, and ``controller.diagnostics()`` for exo_phase_valid."""

    side: str
    actuator_id: int
    controller: DLLeg
    front_end: BootSensorFrontEnd


class ExoBootDLDevice:
    """The DL task on both legs at a fixed rate inside the physics loop; a ``DeviceController``."""

    def __init__(
        self,
        legs: Sequence[DLLegDrive],
        *,
        weights,
        schedule: TickSchedule,
        latency_ticks: int = 1,
        dtype=np.float32,
    ):
        if latency_ticks < 1:
            raise ValueError(
                f"latency_ticks must be >= 1 (the reply to a sample cannot arrive the same tick), got {latency_ticks}"
            )
        self.legs = list(legs)
        self.actuator_ids = tuple(leg.actuator_id for leg in self.legs)
        self.schedule = schedule
        self.latency_ticks = latency_ticks
        self.net = StreamingGaitNet(weights, n_streams=len(self.legs), heads=NET_HEADS, dtype=dtype)
        self._samples = np.zeros((len(self.legs), len(INPUT_CHANNELS)), dtype=dtype)
        self.reset()

    def reset(self) -> None:
        self.schedule.reset()
        self.net.reset()
        for leg in self.legs:
            leg.front_end.reset()
            leg.controller.reset()
        self._replies: deque = deque(maxlen=self.latency_ticks)
        self._torques = [0.0] * len(self.legs)

    def compute_torque(self, sim) -> Sequence[float]:
        if not self.schedule.tick():
            return self._torques
        t = float(sim.data.time)
        # The reply each leg uses now was computed latency_ticks ago, from the buffer as it was then.
        reply = self._replies[0] if len(self._replies) == self.latency_ticks else None
        for i, leg in enumerate(self.legs):
            sent = [round(v, 5) for v in leg.front_end.sample(sim)]  # '%.5f' on the Pi
            self._samples[i] = sent
            leg.controller.sent = sent
        out = self.net.step(self._samples)
        self._replies.append(
            [
                (
                    round(float(out["stance_phase"][i]), 5),
                    float(np.round(out["stance_swing"][i])),
                    round(float(out["velocity"][i]), 5),
                )
                for i in range(len(self.legs))
            ]
        )
        torques = []
        for i, leg in enumerate(self.legs):
            torque = leg.controller.step(t, None if reply is None else reply[i])
            torques.append(PLANTARFLEXION_SIGN * torque)
        self._torques = torques
        return self._torques


def repo_path(path: str) -> pathlib.Path:
    """A config's file path: as given if absolute, else relative to the repo root (not the working directory)."""
    path = pathlib.Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def build_dl_device(params, model: mujoco.MjModel, *, physics_rate_hz: float) -> ExoBootDLDevice:
    """The boot's DL task at 175 Hz, from the simulated boot sensors, which the model must carry
    (``boot_sensors.add_boot_imus``). ``params.controller_rate_hz`` must be 175. With
    ``params.dl_out_of_plane_filter_path``, each leg's out-of-plane IMU channels are synthesized by that file's filter."""
    if params.controller_rate_hz != DL_RATE_HZ:
        raise ValueError(
            f"device_controller 'exoboot_dl' runs at {DL_RATE_HZ:g} Hz, got controller_rate_hz="
            f"{params.controller_rate_hz:g}: its network was trained on the boot's 175 Hz samples, and its 200-sample "
            "window and speed filter count them"
        )
    schedule = TickSchedule(rate_hz=params.controller_rate_hz, physics_rate_hz=physics_rate_hz)
    weights, _ = load_weights(repo_path(params.dl_weights_path))
    filters = {}
    if params.dl_out_of_plane_filter_path:
        filters = load_out_of_plane_filters(repo_path(params.dl_out_of_plane_filter_path))
        if set(filters) != {"r", "l"}:
            raise ValueError(f"{params.dl_out_of_plane_filter_path} has filters for {sorted(filters)}; need 'r' and 'l'")
    spline_params = dict(
        rise_fraction=params.rise_fraction,
        peak_fraction=params.peak_fraction,
        fall_fraction=params.fall_fraction,
        peak_torque=params.peak_torque,
        bias_torque=params.bias_torque,
        peak_hold_time=params.peak_hold_time,
    )
    legs = []
    for side, actuator in (("r", params.right_actuator), ("l", params.left_actuator)):
        actuator_id = ankle_torque_actuator(model, actuator, side)  # this side's ankle torque actuator, or it refuses
        front_end = BootSensorFrontEnd(
            model,
            side,
            standing_angle_deg=getattr(params, f"ankle_standing_angle_{side}_deg"),
            sample_rate_hz=params.controller_rate_hz,
            mount_rpy_deg=tuple(getattr(params, f"imu_mount_{side}_{axis}_deg") for axis in ("roll", "pitch", "yaw")),
            quantize=params.sensor_quantize,
            out_of_plane=filters.get(side),
        )
        leg = DLLeg(
            spline=FourPointSpline(**spline_params),
            state_machine=BootStateMachine(reel_in_time=params.reel_in_time, reel_out_time=params.reel_out_time),
            assist_on=params.dl_assist_on,
            speed=SpeedActivation(on_speed=params.dl_speed_on, off_speed=params.dl_speed_off),
        )
        legs.append(DLLegDrive(side=side, actuator_id=actuator_id, controller=leg, front_end=front_end))
    return ExoBootDLDevice(
        legs,
        weights=weights,
        schedule=schedule,
        latency_ticks=params.dl_latency_ticks,
        dtype=np.float32 if params.dl_float32 else np.float64,
    )
