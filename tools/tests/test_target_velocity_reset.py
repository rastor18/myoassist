"""A training reset draws the target velocity from the config's band, and leaves the band alone.

`_change_mode_and_target_velocity_randomly` runs at construction and on every training reset, and
hands the episode's speed profile to `set_target_velocity_mode_manually`, which also overwrites
`_min_target_velocity` / `_max_target_velocity` from its arguments. From 4a4cbe3 (2025-08-05) the
call passed them in the wrong order: the random phase in [0, 2*pi] became the new max and the old
max the new min, so after a reset or two the target speed lay between two random numbers in
[0, 2*pi] m/s whatever the config said. Nothing raised, and evaluation runs in evaluate mode, which
skips the randomiser, so no rollout showed it.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from tools.tests.conftest import REPO_ROOT, build_env, shipped_configs

CONFIGS = [
    # A flat 1.25 m/s, as every shipped device config asks for.
    next(p for p in shipped_configs() if "Tutorial_L1" in p.name),
    # A real band, so the UNIFORM, SINUSOIDAL and STEP draws are exercised inside it.
    REPO_ROOT / "rl_train/train/train_configs/imitation_tutorial_22_separated_net_speed_control.json",
]


def _unwrap(env):
    return (env.envs[0] if hasattr(env, "envs") else env).unwrapped


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.stem[:40])
def test_training_resets_keep_the_target_velocity_in_the_configured_band(config_path):
    env_params = json.loads(config_path.read_text())["env_params"]
    lo, hi = env_params["min_target_velocity"], env_params["max_target_velocity"]
    # The randomiser draws from both global generators; seed them so a failure reproduces.
    random.seed(0)
    np.random.seed(0)

    _, env = build_env(config_path)
    try:
        u = _unwrap(env)
        assert not u.is_evaluate_mode, "evaluate mode skips the randomiser, so this test would prove nothing"
        zero_action = np.zeros(env.action_space.shape)
        modes = set()
        for episode in range(12):
            env.reset()
            modes.add(u._velocity_mode_for_this_episode)
            # The band itself must survive the reset: the bug overwrote it with the random phase.
            assert (u._min_target_velocity, u._max_target_velocity) == (lo, hi), (
                f"{config_path.name}: reset {episode} moved the band from [{lo}, {hi}] to "
                f"[{u._min_target_velocity:.2f}, {u._max_target_velocity:.2f}] m/s"
            )
            # A few steps as well, since SINUSOIDAL re-reads the band every step.
            for step in range(4):
                assert lo - 1e-9 <= u._target_velocity <= hi + 1e-9, (
                    f"{config_path.name}: reset {episode}, step {step} ({u._velocity_mode_for_this_episode.name}) "
                    f"targets {u._target_velocity:.2f} m/s, outside the config's [{lo}, {hi}]"
                )
                env.step(zero_action)
        assert len(modes) > 1, f"12 resets drew only {modes}; the band check covered one velocity mode"
    finally:
        env.close()
