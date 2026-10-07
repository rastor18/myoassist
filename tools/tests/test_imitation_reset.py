"""MyoAssistLegImitation.reset: an episode starts from its reference index alone, with every joint equality satisfied.

The reference sets only its own joints (hips, knees, ankles, pelvis). Every other joint used to keep the position and
velocity the previous episode ended with: the toes, and the 28 knee translations and muscle via points that joint
equalities tie to the knee and hip angles. So two resets to the same index never started alike, and each episode began
with those constraints violated, by up to about 6 cm, for the solver to snap back on the first step.

The equalities are checked with MuJoCo's own constraint residuals rather than a re-derivation of the polynomials, so the
test does not share the reset's arithmetic. The in-loop device env inherits this reset;
tools/tests/test_rollout_controllers.py covers it there through ``reset_at``.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from tools.tests.conftest import DEVICE_SWEEP_DIR

# myolegs22 + DephyExoBoot_L1 on the stock exo env. The reset under test is MyoAssistLegImitation's, which every
# imitation env inherits.
CONFIG = DEVICE_SWEEP_DIR / "imitation_22_DephyExoBoot_L1_h128_e32_sidenet_mirror0p1_actpen10.json"


@pytest.fixture(scope="module")
def env():
    from rl_train.envs.environment_handler import EnvironmentHandler
    from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
    from rl_train.utils.data_types import DictionableDataclass

    config = DictionableDataclass.create(ExoImitationTrainSessionConfig, json.loads(CONFIG.read_text()))
    config.env_params.num_envs = 1
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False, is_evaluate_mode=True)
    # The target speed scales the reference velocities a reset applies, and a training reset redraws it. Hold it at one
    # value, so that only the reference index and what ran before can differ between resets.
    speed = config.env_params.min_target_velocity
    env.set_target_velocity_mode_manually(
        mode=type(env).VelocityMode.UNIFORM,
        starting_phase=0.0,
        initial_target_velocity=speed,
        min_target_velocity=speed,
        max_target_velocity=speed,
    )
    yield env
    env.close()


def _rollout(env, rng, steps):
    """Random actions: they leave every joint, the reference's or not, somewhere the reset has to undo."""
    for _ in range(steps):
        *_, terminated, truncated, _ = env.step(rng.uniform(-1.0, 1.0, env.action_space.shape[0]))
        if terminated or truncated:
            break


def _unreferenced_qpos(env):
    """qpos addresses of the joints the reference does not set: the toes and the joints tied to the knees and hips."""
    model = env.sim.model
    referenced = {int(model.jnt_qposadr[model.joint(key).id]) for key in env.reference_data_keys}
    return np.array([i for i in range(model.nq) if i not in referenced])


def _joint_equality_residuals(env):
    """Position and velocity residual of every joint-equality row, as MuJoCo's constraint solver sees them."""
    model = getattr(env.sim.model, "ptr", env.sim.model)  # dm_control's wrapper -> the raw MjModel
    data = getattr(env.sim.data, "ptr", env.sim.data)
    mujoco.mj_forward(model, data)
    rows = np.flatnonzero(data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    rows = rows[model.eq_type[data.efc_id[rows]] == mujoco.mjtEq.mjEQ_JOINT]
    return data.efc_pos[rows], data.efc_vel[rows]


def test_two_resets_to_one_index_start_alike_whatever_ran_before(env, monkeypatch):
    monkeypatch.setattr(env, "_flag_random_ref_index", False)  # every reset starts at reference index 0
    rng = np.random.default_rng(0)
    others = _unreferenced_qpos(env)

    env.reset()
    first = env.sim.data.qpos.copy(), env.sim.data.qvel.copy()
    # Before each further reset: a rollout from index 0, an episode from a random index, and nothing at all.
    for steps, random_start in ((25, False), (10, True), (0, False)):
        if random_start:
            env._flag_random_ref_index = True
            env.reset()
            env._flag_random_ref_index = False
        _rollout(env, rng, steps)
        if steps:
            assert not np.allclose(env.sim.data.qpos[others], first[0][others]), (
                "the rollout left the joints the reference does not set where the reset puts them; this proves nothing"
            )
        env.reset()
        np.testing.assert_array_equal(env.sim.data.qpos, first[0])
        np.testing.assert_array_equal(env.sim.data.qvel, first[1])


def test_every_joint_equality_holds_right_after_reset(env):
    assert env._flag_random_ref_index, "the config draws each episode's index, as training does"
    model = getattr(env.sim.model, "ptr", env.sim.model)
    joint_equalities = np.count_nonzero((model.eq_type == mujoco.mjtEq.mjEQ_JOINT) & model.eq_active0)
    assert joint_equalities > 0
    rng = np.random.default_rng(1)
    for _ in range(4):
        _rollout(env, rng, 20)
        env.reset()
        pos, vel = _joint_equality_residuals(env)
        assert len(pos) == joint_equalities, "every active joint equality is a constraint row"
        np.testing.assert_allclose(pos, 0.0, atol=1e-10, err_msg=f"position residuals at index {env._imitation_index}")
        np.testing.assert_allclose(vel, 0.0, atol=1e-10, err_msg=f"velocity residuals at index {env._imitation_index}")
