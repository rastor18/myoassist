"""Device controllers that run inside the physics loop, once per physics substep.

myosuite's ``MujocoEnv.step`` writes ``data.ctrl`` once and then advances every physics substep of the control
step in a single call (``robot.step`` -> ``sim.advance(n)``), so nothing can act between substeps. A device
controller that has to run faster than the control rate -- the ExoBoot's own 175 Hz loop, or a VNMC with 10 ms
activation dynamics -- needs that seam. ``DeviceControlledMujocoEnv.step`` provides it by doing what the stock
step does, split up: the action is clipped and processed once, exactly as ``robot.step`` processes it, and then
on each substep the controller sets its actuators' ctrl before the sim advances by that one substep.

The env stays stock until a controller is installed with ``set_device_controller``. MyoAssist's setup steps the
env before then (myosuite's probe step and the settle steps), and those take the stock path.

Python places this class below ``MyoAssistLegBase``, e.g.
``class Env(MyoAssistLegImitationExo, DeviceControlledMujocoEnv)``, so ``MyoAssistLegBase.step``'s
``super().step()`` lands here and every per-step piece of MyoAssist's own logic runs around it unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import mujoco
import numpy as np
from myosuite.envs import env_base

from myoassist_utils.exo_ctrl.torque_adapter import torque_actuator_params


class DeviceController(Protocol):
    """Drives some of the model's actuators from inside the physics loop.

    ``compute_torque`` is called once per physics substep, before the sim advances by that substep. It returns one
    joint torque in N*m per entry of ``actuator_ids``, in the joint's own sign convention: the generalized force the
    actuator puts on its joint, MuJoCo's ``actuator_force * gear``. The env turns each into ctrl and clips it to the
    actuator's ctrlrange. A controller slower than the physics keeps its own schedule and repeats its held torques
    in between.

    ``reset`` is called at the start of every episode, after the sim has been reset, so the episode's initial state
    can be read there or on the first ``compute_torque``.
    """

    actuator_ids: Sequence[int]

    def reset(self) -> None: ...

    def compute_torque(self, sim) -> Sequence[float]: ...


class JointTorqueDrive:
    """Joint torque -> ctrl for one fixed-gain, bias-free, dynamics-free joint actuator, in plain floats."""

    __slots__ = ("actuator_id", "torque_per_ctrl", "ctrl_low", "ctrl_high")

    def __init__(self, model: mujoco.MjModel, actuator_id: int):
        gain, gear, ctrl_low, ctrl_high = torque_actuator_params(model, actuator_id)
        if gain * gear == 0 or not np.isfinite(gain * gear):
            raise ValueError(f"actuator {actuator_id}: gain * gear must be finite and non-zero, got {gain} * {gear}")
        self.actuator_id = int(actuator_id)
        self.torque_per_ctrl = gain * gear
        self.ctrl_low = ctrl_low
        self.ctrl_high = ctrl_high

    def ctrl(self, torque: float) -> float:
        """The ctrl that makes this actuator apply ``torque`` to its joint, saturated at the ctrlrange."""
        ctrl = torque / self.torque_per_ctrl
        if ctrl < self.ctrl_low:
            return self.ctrl_low
        if ctrl > self.ctrl_high:
            return self.ctrl_high
        if ctrl != ctrl:
            # NaN passes both bounds. In data.ctrl it would make MuJoCo reset the state mid-episode, with a warning
            # at most, and every sample after it would be garbage.
            raise ValueError(f"device controller returned a NaN torque for actuator {self.actuator_id}")
        return ctrl


class DeviceControlledMujocoEnv(env_base.MujocoEnv):
    """A ``MujocoEnv`` whose ``step`` runs an installed ``DeviceController`` on every physics substep."""

    _device_controller: DeviceController | None = None
    _device_drives: tuple[JointTorqueDrive, ...] = ()

    @property
    def device_controller(self) -> DeviceController | None:
        return self._device_controller

    def set_device_controller(self, controller: DeviceController | None) -> None:
        """Install ``controller`` (``None`` returns the env to the stock step). It is reset here, and on every reset."""
        if controller is None:
            self._device_controller, self._device_drives = None, ()
            return
        if self.robot.is_hardware:
            raise RuntimeError("device controllers drive the simulation; this robot is hardware")
        ids = [int(i) for i in controller.actuator_ids]
        if len(set(ids)) != len(ids):
            raise ValueError(f"device controller lists an actuator twice: {ids}")
        model = getattr(self.sim.model, "ptr", self.sim.model)  # dm_control's wrapper -> the raw MjModel
        self._device_drives = tuple(JointTorqueDrive(model, i) for i in ids)
        self._device_controller = controller
        controller.reset()

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        if self._device_controller is not None:
            self._device_controller.reset()
        return obs

    def step(self, a, **kwargs):
        controller = self._device_controller
        if controller is None:
            return super().step(a, **kwargs)

        # MujocoEnv.step with robot.step inlined and its sim.advance(n) split into n single substeps. Everything else is
        # the stock code: the same clip, the same actuator processing, the same substep count, and forward() at the end.
        a = np.clip(a, self.action_space.low, self.action_space.high)
        ctrl = self.robot.process_actuator(
            controls=a,
            step_duration=self.dt,
            normalized=self.normalize_act,
            position_limits=True,
            velocity_limits=True,
            out_space="sim",
        )
        sim = self.sim
        data_ctrl = sim.data.ctrl
        data_ctrl[:] = ctrl
        drives = self._device_drives
        for _ in range(int(self.dt / sim.step_duration)):
            for drive, torque in zip(drives, controller.compute_torque(sim), strict=True):
                data_ctrl[drive.actuator_id] = drive.ctrl(torque)
            sim.advance(substeps=1, render=False)
        # Stock rendering draws once per control step, after the substeps, and so does this. It does not throttle to
        # real time as robot.step does when rendering on screen.
        if self.mujoco_render_frames:
            sim.renderer.render_to_window()
        self.last_ctrl = data_ctrl.copy()
        return self.forward(**kwargs)
