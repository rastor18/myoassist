"""Exo controllers for the in-loop seam (``rl_train.envs.device_control``), which calls them every physics substep.

They satisfy ``rl_train.envs.device_control.DeviceController`` structurally -- ``actuator_ids``, ``reset()``,
``compute_torque(sim)`` -- so this package still does not import rl_train. That seam takes joint torques in MuJoCo's
convention, so the plantarflexion-positive torques of the ExoBoot controllers are flipped to the ankle's
dorsiflexion-positive sign here, and nowhere else on this path.

Per-substep code stays in plain Python floats: NumPy on scalars costs microseconds per operation, and this runs 1200
times per simulated second in every env worker.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import mujoco

from myoassist_utils.exo_ctrl.dl_controller import build_dl_device
from myoassist_utils.exo_ctrl.factory import LegExo, build_leg_exos
from myoassist_utils.exo_ctrl.schedule import TickSchedule
from myoassist_utils.exo_ctrl.torque_adapter import PLANTARFLEXION_SIGN

# Device controllers that read the simulated boot sensors, which the model has to be composed with.
NEEDS_BOOT_SENSORS = frozenset({"exoboot_dl"})


class ZeroTorqueDevice:
    """No torque on the given actuators, on every substep.

    The in-loop seam with the exo off. With it the device env must reproduce the stock env exactly, which is the test
    that the split-up step is the stock step; and the throughput benchmark uses it to price the loop alone.
    """

    def __init__(self, actuator_ids: Sequence[int]):
        self.actuator_ids = tuple(int(i) for i in actuator_ids)
        self._torques = (0.0,) * len(self.actuator_ids)

    def reset(self) -> None:
        pass

    def compute_torque(self, sim) -> Sequence[float]:
        return self._torques


class FootForce:
    """One leg's foot + toes touch sensors: the sum ``MyoAssistLegBase._get_foot_force`` takes, read from the raw array."""

    def __init__(self, model: mujoco.MjModel, side: str):
        adrs = []
        for part in ("foot", "toes"):
            name = f"{side}_{part}"
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sensor_id < 0:
                raise KeyError(f"no sensor named {name!r} in the model")
            if model.sensor_dim[sensor_id] != 1:
                raise ValueError(f"sensor {name!r} has dim {model.sensor_dim[sensor_id]}; a touch sensor has 1")
            adrs.append(int(model.sensor_adr[sensor_id]))
        self._foot, self._toes = adrs

    def __call__(self, sim) -> float:
        sensordata = sim.data.sensordata
        return float(sensordata[self._foot] + sensordata[self._toes])


class FixedRateLegExos:
    """Per-leg ``LegExoController``s at a fixed rate inside the physics loop, holding their torques between ticks.

    This is the boot's own arrangement. Its main loop reads the sensors, runs both sides' controllers and sends each motor
    a new command (at 175 Hz on the boot), and the motors hold that command until the next one. On each tick here, every
    leg steps its controller with the current sim time and its own signal. Between ticks the last torques are returned
    unchanged.
    """

    def __init__(
        self,
        legs: Sequence[LegExo],
        *,
        signals: Sequence[Callable[[object], float]],
        schedule: TickSchedule,
    ):
        if len(signals) != len(legs):
            raise ValueError(f"need one signal per leg, got {len(signals)} for {len(legs)} legs")
        self.legs = list(legs)
        self.actuator_ids = tuple(leg.actuator_id for leg in self.legs)
        self.schedule = schedule
        self._signals = tuple(signals)
        self._torques = [0.0] * len(self.legs)

    def reset(self) -> None:
        self.schedule.reset()
        for leg in self.legs:
            leg.controller.reset()
        self._torques = [0.0] * len(self.legs)

    def compute_torque(self, sim) -> Sequence[float]:
        if self.schedule.tick():
            t = float(sim.data.time)
            self._torques = [
                PLANTARFLEXION_SIGN * leg.controller.step(t, signal(sim)) for leg, signal in zip(self.legs, self._signals)
            ]
        return self._torques


class ShadowDevice:
    """Runs another device controller on every substep exactly as it would run, and applies none of its torque.

    For judging a controller's sensing on a policy that walks without it: the gait is the exo-off gait bit for bit, and
    the controller's ``diagnostics()`` still report what it would do. Its ``legs`` and ``schedule`` are passed through.
    """

    def __init__(self, inner):
        self.inner = inner
        self.actuator_ids = tuple(inner.actuator_ids)
        self.legs = list(getattr(inner, "legs", []))
        self.schedule = getattr(inner, "schedule", None)
        self._torques = (0.0,) * len(self.actuator_ids)

    def reset(self) -> None:
        self.inner.reset()

    def compute_torque(self, sim) -> Sequence[float]:
        self.inner.compute_torque(sim)
        return self._torques


def _actuator_id(model: mujoco.MjModel, name: str) -> int:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id < 0:
        raise KeyError(f"no actuator named {name!r} in the model")
    return actuator_id


def _build_zero(params, model: mujoco.MjModel, *, physics_rate_hz: float) -> ZeroTorqueDevice:
    """No torque on the two exo actuators."""
    return ZeroTorqueDevice([_actuator_id(model, params.right_actuator), _actuator_id(model, params.left_actuator)])


def _build_spline(params, model: mujoco.MjModel, *, physics_rate_hz: float) -> FixedRateLegExos:
    """The ExoBoot four-point spline (the boot's WALKING task) at ``params.controller_rate_hz`` (150 Hz by default, every
    8th substep of 1200 Hz physics), heel strikes from foot GRF. It ticks every 6.7 ms against the boot's 5.7 ms; on a
    boot log that delays the torque by about half a tick (``tools/replay_exoboot_log.py`` layer 6 measures it)."""
    schedule = TickSchedule(rate_hz=params.controller_rate_hz, physics_rate_hz=physics_rate_hz)
    legs = build_leg_exos(params, model)
    return FixedRateLegExos(legs, signals=[FootForce(model, leg.side) for leg in legs], schedule=schedule)


# Each in-loop controller by its ``env_params.device_controller`` name: a builder taking ``ExoControllerParams``-shaped
# params, the composed model and the physics rate.
DEVICE_BUILDERS: dict[str, Callable[..., object]] = {
    "zero": _build_zero,
    "exoboot_spline": _build_spline,
    "exoboot_dl": build_dl_device,
}
DEVICE_CONTROLLERS = tuple(DEVICE_BUILDERS)


def build_device_controller(name: str, params, model: mujoco.MjModel, *, physics_rate_hz: float):
    """The in-loop controller ``name`` drives (one of ``DEVICE_BUILDERS``, whose docstrings say what each is), built
    from ``ExoControllerParams``-shaped ``params``. With ``params.shadow_mode`` it runs and applies no torque
    (``ShadowDevice``)."""
    if name not in DEVICE_BUILDERS:
        raise ValueError(f"unknown device controller {name!r}; expected one of {DEVICE_CONTROLLERS}")
    controller = DEVICE_BUILDERS[name](params, model, physics_rate_hz=physics_rate_hz)
    return ShadowDevice(controller) if getattr(params, "shadow_mode", False) else controller
