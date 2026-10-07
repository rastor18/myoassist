from myoassist_utils.exo_ctrl import build_device_controller
from rl_train.envs.device_control import DeviceControlledMujocoEnv
from rl_train.envs.exo_controller_mixin import ExoControllerMixin
from rl_train.envs.myoassist_leg_imitation import MyoAssistLegImitation


class MyoAssistLegImitationExo(ExoControllerMixin, MyoAssistLegImitation):
    # override
    def step(self, a, **kwargs):
        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)

        return (next_obs, reward, terminated, truncated, info)


class MyoAssistLegImitationExoDevice(MyoAssistLegImitationExo, DeviceControlledMujocoEnv):
    """``MyoAssistLegImitationExo`` with ``env_params.device_controller`` driving the exo inside the physics loop.

    The controller is installed after setup, so setup's own steps are stock. With no controller named, this is
    ``MyoAssistLegImitationExo`` exactly.
    """

    def _setup(self, *, env_params, **kwargs):
        name = getattr(env_params, "device_controller", "")
        params = getattr(env_params, "exo_controller_params", None)
        if name and params is None:
            raise ValueError(
                f"device_controller={name!r} reads its parameters from exo_controller_params, which this config "
                f"({type(env_params).__qualname__}) does not have; load it as ExoImitationTrainSessionConfig"
            )
        super()._setup(env_params=env_params, **kwargs)
        if name:
            model = getattr(self.sim.model, "ptr", self.sim.model)  # dm_control's wrapper -> the raw MjModel
            controller = build_device_controller(name, params, model, physics_rate_hz=env_params.physics_sim_framerate)
            self.set_device_controller(controller)
            self._exo_diagnostic_legs = getattr(controller, "legs", [])
