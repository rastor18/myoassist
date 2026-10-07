"""Checks and the training diagnostic for a scripted exo controller (``env_params.device_controller``).

The controller itself runs inside the physics loop, on every substep: see ``device_control.py`` and the env id
``myoAssistLegImitationExoDevice-v0``. This mixin sits on the stock exo env too, so that a config naming a device
controller on an env that cannot run one fails at build instead of silently training without it, and so that the
``exo_phase_valid`` diagnostic is emitted wherever a config weights it (the exo-off baseline config carries it too).

The policy still emits the exo actions; the device controller overwrites the exo's ctrl. Configs also pin those actions
with the exo-off pattern (a ``range_mapping`` then a ``constant`` of 1.0 on the exo indices), so the two policy outputs
never matter.
"""

from __future__ import annotations

import warnings

import numpy as np

from rl_train.envs.device_control import DeviceControlledMujocoEnv


class ExoControllerMixin:
    EXO_PHASE_VALID_KEY = "exo_phase_valid"

    def _setup(self, *, env_params, **kwargs):
        device_controller = getattr(env_params, "device_controller", "")
        if device_controller and not isinstance(self, DeviceControlledMujocoEnv):
            raise ValueError(
                f"device_controller={device_controller!r} needs the env id myoAssistLegImitationExoDevice-v0; "
                f"{type(self).__name__} has no in-loop seam and would silently run without it"
            )
        if device_controller:
            weights = env_params.reward_keys_and_weights
            if getattr(weights, "exo_activation_penalty", 0.0):
                warnings.warn(
                    "exo_activation_penalty is non-zero while a scripted exo controller drives the exo; it prices "
                    "torque the policy does not choose, and only adds a phase-locked offset to the reward",
                    stacklevel=2,
                )
            if getattr(weights, self.EXO_PHASE_VALID_KEY, 0.0):
                warnings.warn(
                    f"{self.EXO_PHASE_VALID_KEY} has a non-zero weight; it is a diagnostic, and weighting it "
                    "rewards the policy for keeping the controller's gait phase valid",
                    stacklevel=2,
                )
        # Whose diagnostics exo_phase_valid reports: the device env points this at its controller's legs.
        self._exo_diagnostic_legs = []
        super()._setup(env_params=env_params, **kwargs)

    def get_reward_dict(self, obs_dict):
        rwd_dict = super().get_reward_dict(obs_dict)
        # Emitted exactly when the config has a weight for it, because _setup asserts the two key sets are equal.
        weight = getattr(self, "rwd_keys_wt", {}).get(self.EXO_PHASE_VALID_KEY)
        if weight is not None:
            legs = getattr(self, "_exo_diagnostic_legs", [])
            valid = float(np.mean([leg.controller.diagnostics()["phase_valid"] for leg in legs])) if legs else 0.0
            rwd_dict[self.EXO_PHASE_VALID_KEY] = valid
            # dense was summed before this key existed, so add its term here or a non-zero weight is silently lost.
            rwd_dict["dense"] = rwd_dict["dense"] + weight * valid
        return rwd_dict
