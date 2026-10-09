"""An episode's start state depends only on its reference index and its target velocity.

`MyoAssistLegImitation.reset` used to write only the reference joints and hand the rest of
`sim.data` to the simulation reset as it stood, so every other DOF started where the previous
episode's fall left it, the joints that equality constraints tie to the knees and hips started off
those constraints, and the reference velocities were scaled by the previous episode's target. The
target-velocity schedule was also read at the previous episode's clock.
"""

from __future__ import annotations

import json
import pathlib

import mujoco
import numpy as np
import pytest

from tools.tests.conftest import REPO_ROOT

TRAIN_CONFIGS = REPO_ROOT / "rl_train/train/train_configs"
SPEED_CONTROL_CONFIG = TRAIN_CONFIGS / "imitation_tutorial_22_separated_net_speed_control.json"


def _shipped_configs() -> list:
    paths = (
        sorted((TRAIN_CONFIGS / "device_sweep").glob("imitation_22_*.json"))
        + sorted((TRAIN_CONFIGS / "prosthesis").glob("imitation_22_*.json"))
        + sorted(TRAIN_CONFIGS.glob("imitation_tutorial_22_*.json"))
    )
    assert paths, f"no shipped configs under {TRAIN_CONFIGS}"
    # assist_sim places UTAnkleExo_L2's free-floating exo for myolegs26's frame, and MuJoCo
    # derives each connect constraint's leg-side anchor from qpos0, where myolegs22 hangs 0.9 m
    # lower. Its constraints therefore hold the exo about 1 m from the foot and cannot all hold
    # at a walking pose, so reset refuses to start an episode on it.
    broken = pytest.mark.xfail(
        reason="UTAnkleExo_L2 connect anchors are 0.9 m off on myolegs22 (assist_sim)", raises=AssertionError, strict=True
    )
    return [pytest.param(p, marks=broken) if "UTAnkleExo_L2" in p.name else p for p in paths]


def _make_env(path: pathlib.Path, *, is_evaluate_mode: bool):
    from rl_train.envs.environment_handler import EnvironmentHandler
    from rl_train.utils.data_types import DictionableDataclass

    raw = json.loads(path.read_text())
    config = DictionableDataclass.create(EnvironmentHandler.get_config_type_from_session_id(raw["env_params"]["env_id"]), raw)
    config.env_params.num_envs = 1
    return EnvironmentHandler.create_environment(config, is_rendering_on=False, is_evaluate_mode=is_evaluate_mode)


def _equality_residuals(env) -> tuple[float, float]:
    """MuJoCo's own position and velocity residual over every equality constraint."""
    model, data = env.sim.model.ptr, env.sim.data.ptr
    mujoco.mj_fwdPosition(model, data)
    rows = data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    jacobian = data.efc_J[: data.nefc * model.nv].reshape(data.nefc, model.nv)[rows]
    return float(np.max(np.abs(data.efc_pos[: data.nefc][rows]))), float(np.max(np.abs(jacobian @ data.qvel)))


@pytest.mark.parametrize("path", _shipped_configs(), ids=lambda p: p.stem.split("_h128")[0])
def test_start_state_is_independent_of_history_and_on_the_constraints(path):
    from rl_train.envs.myoassist_leg_base import MyoAssistLegBase

    env = _make_env(path, is_evaluate_mode=True)
    try:
        env.set_target_velocity_mode_manually(
            mode=MyoAssistLegBase.VelocityMode.UNIFORM,
            starting_phase=0.0,
            initial_target_velocity=1.25,
            min_target_velocity=1.25,
            max_target_velocity=1.25,
        )
        env._flag_random_ref_index = False
        rng = np.random.default_rng(0)
        starts = []
        for history in (0, 25, 60):
            for _ in range(history):
                env.step(rng.uniform(-1, 1, env.sim.model.nu))
            env.reset()
            starts.append((env.sim.data.qpos.copy(), env.sim.data.qvel.copy(), env.sim.data.act.copy()))
            position_residual, velocity_residual = _equality_residuals(env)
            assert position_residual < 1e-9 and velocity_residual < 1e-9, (
                f"after {history} steps of history, reset starts the equality constraints off by "
                f"{position_residual:.3g} in position and {velocity_residual:.3g} in velocity"
            )
        for (qpos, qvel, act), history in zip(starts[1:], (25, 60)):
            changed = np.flatnonzero((qpos != starts[0][0]) | (qvel != starts[0][1]))
            assert not len(changed), (
                f"after {history} steps of history, reset to the same index starts DOFs "
                f"{[env.sim.model.joint(int(env.sim.model.dof_jntid[i])).name for i in changed]} differently"
            )
            assert np.array_equal(act, starts[0][2])
    finally:
        env.close()


def test_reference_velocities_use_this_episodes_target_velocity():
    """The reset pose scales the reference velocities by the target drawn for the new episode."""
    env = _make_env(SPEED_CONTROL_CONFIG, is_evaluate_mode=False)
    try:
        for _ in range(12):
            env.reset()
            series = env._reference_data["series_data"]
            index = env._imitation_index
            ratio = env._target_velocity / series["dq_pelvis_tx"][index]
            for key in env.reference_data_keys:
                expected = series[f"dq_{key}"][index] * ratio
                actual = env.sim.data.joint(key).qvel[0]
                assert np.isclose(actual, expected), (key, actual, expected, env._target_velocity)
    finally:
        env.close()


def test_target_velocity_schedule_starts_on_the_new_episodes_clock():
    """The first step commands the target the reset started at, and step changes count from t = 0."""
    from rl_train.envs.myoassist_leg_base import MyoAssistLegBase

    env = _make_env(SPEED_CONTROL_CONFIG, is_evaluate_mode=False)
    try:
        rng = np.random.default_rng(0)
        modes = set()
        for _ in range(30):
            for _ in range(20):
                env.step(rng.uniform(-1, 1, env.sim.model.nu))
            env.reset()
            modes.add(env._velocity_mode_for_this_episode)
            assert env.sim.data.time == 0.0
            assert env._prev_step_changed_time == 0.0
            at_reset = env._target_velocity
            env.step(np.zeros(env.sim.model.nu))
            # The first step reads the schedule at t = 0 before advancing the clock, so for every
            # mode it must command exactly the target the start state was built for.
            assert env._target_velocity == at_reset, (env._velocity_mode_for_this_episode, at_reset, env._target_velocity)
        assert MyoAssistLegBase.VelocityMode.SINUSOIDAL in modes
    finally:
        env.close()
