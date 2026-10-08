"""The VNMC as the device env runs it (device_controller "exoboot_vnmc"), and its shipped configs.

As for 4PTS (tools/tests/test_device_control.py): the configs are the exo-off experiment but for the controller; in
shadow mode the gait is the stock exo-off gait bit for bit; each tick reads the leg's foot force and its boot's ankle
angle; the VNMC's command is what reaches the ankle, held between ticks; and every episode starts the controller over.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from myoassist_utils.exo_ctrl import TickSchedule
from myoassist_utils.exo_ctrl.boot_state import REEL_OUT, STANCE
from tools.tests.conftest import REPO_ROOT
from tools.tests.test_device_control import (
    DEVICE_ENV_ID,
    EXO_OFF,
    GATE_STEPS,
    _config,
    _exo_ids,
    _joint_torque,
    _make,
    _rollout,
)

CONFIG_DIR = REPO_ROOT / "rl_train/train/train_configs/exoboot_vnmc"
VNMC = CONFIG_DIR / "imitation_22_DephyExoBoot_L1_exoboot_vnmc.json"
SHADOW = CONFIG_DIR / "imitation_22_DephyExoBoot_L1_exoboot_vnmc_shadow.json"


def test_vnmc_configs_are_the_exo_off_experiment_with_the_controller():
    """Same rewards, policy and env as the exo-off baseline; the device env, the VNMC and its own parameters; and the
    shadow config the VNMC config but for shadow_mode."""
    off, on, shadow = (json.loads(p.read_text()) for p in (EXO_OFF, VNMC, SHADOW))
    assert off["env_params"].pop("env_id") == "myoAssistLegImitationExo-v0"
    assert off["env_params"].pop("device_controller", "") == ""
    off["env_params"].pop("exo_controller_params")
    params = {}
    for name, config in (("vnmc", on), ("shadow", shadow)):
        assert config["env_params"].pop("env_id") == DEVICE_ENV_ID, name
        assert config["env_params"].pop("device_controller") == "exoboot_vnmc", name
        params[name] = config["env_params"].pop("exo_controller_params")
        assert config == off, name
    assert params["vnmc"].pop("shadow_mode") is False and params["shadow"].pop("shadow_mode") is True
    assert params["vnmc"] == params["shadow"]
    assert params["vnmc"]["controller_rate_hz"] == 150.0 and params["vnmc"]["vnmc_gain"] == 1.468
    assert params["vnmc"]["reel_out_time"] == 0.2, "the boot's reel-out timer"


@pytest.fixture(scope="module")
def stock_rollout():
    return _rollout(_config(flag_random_ref_index=False), GATE_STEPS)[0]


def test_shadow_vnmc_reproduces_the_stock_step_exactly(stock_rollout):
    """The VNMC in shadow mode runs on every substep, and the gait is the stock exo-off gait bit for bit."""
    shadow, calls = _rollout(_config(SHADOW, flag_random_ref_index=False), GATE_STEPS)
    assert calls == 40 * GATE_STEPS
    for k, (s, d) in enumerate(zip(stock_rollout, shadow, strict=True)):
        for key in ("qpos", "qvel", "act", "obs"):
            assert np.array_equal(s[key], d[key]), f"step {k}: {key} differs by up to {np.max(np.abs(s[key] - d[key]))}"
        assert (s["time"], s["reward"], s["done"]) == (d["time"], d["reward"], d["done"]), f"step {k}"


@pytest.fixture(scope="module")
def vnmc_env():
    env = _make(_config(VNMC, flag_random_ref_index=False))
    yield env
    env.close()


def test_each_tick_reads_the_foot_force_and_the_boots_ankle_angle(vnmc_env, monkeypatch):
    """On the 150 Hz ticks, each leg gets its own foot force and its ankle angle in the boot's degrees: the standing
    angle at the keyframe, plantarflexion-positive, to an encoder click."""
    env = vnmc_env
    standing = {"r": -1.85, "l": -8.47}
    model = env.sim.model
    seen = {leg.side: [] for leg in env.device_controller.legs}
    for leg in env.device_controller.legs:

        def step(t, signal, side=leg.side):
            force, angle = signal
            assert force == env._get_foot_force(side), f"{side}: not the env's foot force"
            q = float(env.sim.data.joint(f"ankle_angle_{side}").qpos[0])
            key_q = float(model.key_qpos[0][model.jnt_qposadr[model.joint(f"ankle_angle_{side}").id]])
            assert angle == pytest.approx(standing[side] - math.degrees(q - key_q), abs=360 / 2**14), side
            seen[side].append(t)
            return 0.0

        monkeypatch.setattr(leg.controller, "step", step)
    env.reset()
    for _ in range(3):
        env.step(env.action_space.sample())
    dt = env.sim.model.opt.timestep
    schedule = TickSchedule(rate_hz=150, physics_rate_hz=1200)
    expected = [n * dt for n in range(3 * 40) if schedule.tick()]
    for side, times in seen.items():
        assert times == pytest.approx(expected, abs=1e-12), side


def test_the_vnmcs_command_reaches_the_ankle_and_is_held_between_ticks(vnmc_env, monkeypatch):
    """With the state machine held in stance, the VNMC's own reflex drives the torque; the joint feels minus that
    command (plantarflexion), and it changes only on ticks."""
    env = vnmc_env
    for leg in env.device_controller.legs:
        machine = leg.controller.state_machine

        def in_stance(t, *, machine=machine, **events):
            machine.state = STANCE
            return STANCE

        monkeypatch.setattr(machine, "step", in_stance)
    env.reset()
    ids = _exo_ids(env)
    felt, commanded = [], []
    for _ in range(30):
        a = env.action_space.sample()
        a[ids] = 1.0  # the policy's own exo action would be zero torque
        env.step(a)
        felt.append([_joint_torque(env, i) for i in ids])
        commanded.append([leg.controller.diagnostics()["torque_nm"] for leg in env.device_controller.legs])
    felt, commanded = np.array(felt), np.array(commanded)
    assert commanded.max() > 1.0, "the reflex built up a torque"
    np.testing.assert_allclose(felt, -commanded, atol=1e-9)


def test_every_episode_starts_the_vnmc_over(vnmc_env):
    """A reset restarts the state machine (reel-out first), the muscle (at rest) and the scaling; two episodes from the
    same start with the same actions report the same diagnostics."""
    env = vnmc_env
    actions = np.random.default_rng(0).uniform(-1.0, 1.0, (20, env.action_space.shape[0]))

    def episode():
        env.reset()
        rows = []
        for a in actions:
            env.step(a)
            rows.append([sorted(leg.controller.diagnostics().items()) for leg in env.device_controller.legs])
        return rows

    first = episode()
    assert first[0][0] != first[-1][0], "the controller moved during the episode"
    again = episode()
    assert first == again
    env.reset()
    for leg in env.device_controller.legs:
        d = leg.controller.diagnostics()
        assert d["control_state"] == REEL_OUT and d["mtu_force"] == 0.0 and d["scalefactor"] == 25.0 / 100
