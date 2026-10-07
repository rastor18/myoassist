"""The in-loop device-controller seam (rl_train/envs/device_control.py) and the exo controllers that use it.

The first test is the gate. With a controller that applies no torque, the split-up step must reproduce the stock step
bit for bit. If it did not, every comparison between the exo-off baseline and an in-loop controller would be confounded
by the step loop itself. The rest check what the stock path has no equivalent for: that a torque set on a substep
reaches the joint on that substep, at the right ctrl, and that a fixed-rate controller runs on its own schedule.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from myoassist_utils.exo_ctrl import TickSchedule
from tools.tests.conftest import REPO_ROOT

CONFIG_DIR = REPO_ROOT / "rl_train/train/train_configs/exoboot_spline"
EXO_OFF = CONFIG_DIR / "imitation_22_DephyExoBoot_L1_exo_off.json"
SPLINE = CONFIG_DIR / "imitation_22_DephyExoBoot_L1_exoboot_spline.json"
DEVICE_ENV_ID = "myoAssistLegImitationExoDevice-v0"
EXO_ACTUATORS = ("Exo_R", "Exo_L")


def _config(path=EXO_OFF, **env_overrides):
    from rl_train.train.train_configs.config_imiatation_exo import (
        ExoImitationTrainSessionConfig,
    )
    from rl_train.utils.data_types import DictionableDataclass

    config = DictionableDataclass.create(ExoImitationTrainSessionConfig, json.loads(path.read_text()))
    config.env_params.num_envs = 1
    for key, value in env_overrides.items():
        setattr(config.env_params, key, value)
    return config


def _make(config):
    from rl_train.envs.environment_handler import EnvironmentHandler

    return EnvironmentHandler.create_environment(config, is_rendering_on=False)


def _exo_ids(env):
    return [env.sim.model.actuator(name).id for name in EXO_ACTUATORS]


def _joint_torque(env, actuator_id):
    """The torque one actuator puts on its joint, as MuJoCo computed it (not qfrc_actuator, which sums the muscles too)."""
    return float(env.sim.data.actuator_force[actuator_id] * env.sim.model.actuator_gear[actuator_id, 0])


class _Constant:
    """The same joint torque on both exo actuators, every substep."""

    def __init__(self, env, torque):
        self.actuator_ids = tuple(_exo_ids(env))
        self._torques = (torque, torque)

    def reset(self):
        pass

    def compute_torque(self, sim):
        return self._torques


def _rollout(config, n_steps: int, seed: int = 0, *, torque: float | None = None):
    """A rollout under a fixed action sequence, with every global RNG the env touches seeded at the same points.

    The reference start index is drawn from an unseeded generator, so callers pin it with flag_random_ref_index=False.
    ``torque`` replaces the configured device controller with a constant one.
    """
    random.seed(seed)
    np.random.seed(seed)
    env = _make(config)
    try:
        if torque is not None:
            env.set_device_controller(_Constant(env, torque))
        random.seed(seed + 1)
        np.random.seed(seed + 1)
        env.reset()
        calls = [0]
        controller = getattr(env, "device_controller", None)
        if controller is not None:
            compute_torque = controller.compute_torque

            def counted(sim):
                calls[0] += 1
                return compute_torque(sim)

            controller.compute_torque = counted
        actions = np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n_steps, env.sim.model.nu))
        actions[:, _exo_ids(env)] = 1.0  # the exo-off pin: ctrl 0 on the stock path
        states = []
        for a in actions:
            obs, reward, terminated, truncated, _ = env.step(a)
            data = env.sim.data
            states.append(
                dict(
                    qpos=data.qpos.copy(),
                    qvel=data.qvel.copy(),
                    act=data.act.copy(),
                    time=float(data.time),
                    obs=np.asarray(obs).copy(),
                    reward=float(reward),
                    done=(bool(terminated), bool(truncated)),
                )
            )
        return states, calls[0]
    finally:
        env.close()


GATE_STEPS = 120


@pytest.fixture(scope="module")
def stock_rollout():
    """The stock exo env under the exo-off config: the baseline the in-loop seam has to reproduce."""
    return _rollout(_config(flag_random_ref_index=False), GATE_STEPS)[0]


def _device_config():
    return _config(flag_random_ref_index=False, env_id=DEVICE_ENV_ID, device_controller="zero")


def test_zero_controller_reproduces_the_stock_step_exactly(stock_rollout):
    """The gate: same seed, same actions, and the device env with no torque matches the stock exo-off env bit for bit."""
    stock = stock_rollout
    device, calls = _rollout(_device_config(), GATE_STEPS)
    assert calls == 40 * GATE_STEPS, f"expected the controller on every one of 40 substeps per step, got {calls} calls"
    # Not a vacuous comparison: the model has moved and time has advanced by the full rollout.
    assert stock[-1]["time"] == pytest.approx(GATE_STEPS / 30, abs=1e-9)
    assert not np.allclose(stock[0]["qpos"], stock[-1]["qpos"])
    for k, (s, d) in enumerate(zip(stock, device, strict=True)):
        for key in ("qpos", "qvel", "act", "obs"):
            assert np.array_equal(s[key], d[key]), f"step {k}: {key} differs by up to {np.max(np.abs(s[key] - d[key]))}"
        assert (s["time"], s["reward"], s["done"]) == (
            d["time"],
            d["reward"],
            d["done"],
        ), f"step {k}"


def test_the_gate_detects_a_micro_newton_metre(stock_rollout):
    """Negative control: the same comparison fails for 1e-6 N*m of plantarflexion, so the pass above is not vacuous."""
    nudged, _ = _rollout(_device_config(), GATE_STEPS, torque=-1e-6)
    differs = [
        k for k, (s, d) in enumerate(zip(stock_rollout, nudged, strict=True)) if not np.array_equal(s["qpos"], d["qpos"])
    ]
    assert differs and differs[0] == 0, f"1e-6 N*m should show from the first step; first difference at {differs[:1]}"


def test_no_device_controller_is_the_stock_env():
    """With device_controller unset the device env installs nothing and its step is MujocoEnv.step itself."""
    env = _make(_config(env_id=DEVICE_ENV_ID))
    try:
        assert env.device_controller is None
    finally:
        env.close()


class _Scripted:
    """Returns the next pair of joint torques on every substep, and records the sim as it was when asked."""

    def __init__(self, env, torques):
        self.actuator_ids = tuple(_exo_ids(env))
        self._env = env
        self._torques = list(torques)
        self.calls = []
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.calls.clear()

    def compute_torque(self, sim):
        seen = tuple(_joint_torque(self._env, i) for i in self.actuator_ids)
        self.calls.append((float(sim.data.time), seen))
        return self._torques[(len(self.calls) - 1) % len(self._torques)]


@pytest.fixture(scope="module")
def device_env():
    env = _make(_config(env_id=DEVICE_ENV_ID, device_controller="zero"))
    yield env
    env.close()


def test_each_substeps_torque_reaches_the_joint_on_that_substep(device_env):
    """Torque set before substep i is what MuJoCo applies during substep i, saturated at the ctrlrange, on each ankle.

    Plantarflexion is negative on these ankles and the exos' ctrlrange is [-1, 0] at gain 100: -150 N*m saturates at
    -100, and +10 (dorsiflexion) clips to 0. The policy's own exo action is -1, which the stock path would make -100.
    """
    env = device_env
    requests = [(-10.0, -20.0), (-30.0, 0.0), (-150.0, 10.0), (-5.0, -45.0)]
    applied = [(-10.0, -20.0), (-30.0, 0.0), (-100.0, 0.0), (-5.0, -45.0)]
    controller = _Scripted(env, requests)
    env.set_device_controller(controller)
    try:
        env.reset()
        t0 = float(env.sim.data.time)
        a = env.action_space.sample()
        a[_exo_ids(env)] = -1.0
        env.step(a)
        assert len(controller.calls) == 40
        dt = env.sim.model.opt.timestep
        for i, (t, _) in enumerate(controller.calls):
            assert t == pytest.approx(t0 + i * dt, abs=1e-12), f"call {i} at t={t}"
        # What the joints felt during substep i is visible at call i + 1, and after the last substep at the end.
        felt = [seen for _, seen in controller.calls[1:]] + [tuple(_joint_torque(env, i) for i in controller.actuator_ids)]
        for i, got in enumerate(felt):
            assert got == pytest.approx(applied[i % len(applied)], abs=1e-9), f"substep {i}"
    finally:
        env.set_device_controller(None)


def test_controller_is_reset_with_every_episode(device_env):
    env = device_env
    controller = _Scripted(env, [(0.0, 0.0)])
    env.set_device_controller(controller)
    try:
        assert controller.resets == 1, "installing a controller resets it"
        env.reset()
        env.reset()
        assert controller.resets == 3
    finally:
        env.set_device_controller(None)


def test_nan_torque_is_refused(device_env):
    env = device_env
    env.set_device_controller(_Scripted(env, [(float("nan"), 0.0)]))
    try:
        env.reset()
        with pytest.raises(ValueError, match="NaN"):
            env.step(env.action_space.sample())
    finally:
        env.set_device_controller(None)


def test_wrong_number_of_torques_is_refused(device_env):
    env = device_env
    env.set_device_controller(_Scripted(env, [(0.0,)]))
    try:
        env.reset()
        with pytest.raises(ValueError):
            env.step(env.action_space.sample())
    finally:
        env.set_device_controller(None)


def test_device_controller_on_the_stock_exo_env_is_refused():
    """Silently ignoring it would train an exo-off policy under a config that names a controller."""
    with pytest.raises(ValueError, match="myoAssistLegImitationExoDevice-v0"):
        _make(_config(device_controller="zero"))


def test_unknown_device_controller_is_refused():
    with pytest.raises(ValueError, match="unknown device controller"):
        _make(_config(env_id=DEVICE_ENV_ID, device_controller="vnmc"))


def test_saved_exo_sessions_load_with_their_controller_fields():
    """Evaluation rebuilds the env from the saved session config, so it has to be read as the session's own class.

    run_policy_eval.py and the in-training analyzer used to read every session as the base imitation class, which has
    no exo controller fields. They vanished without an error, and the evaluation rollout ran with no scripted exo.

    Loading also ignores keys the dataclass does not have, silently, so a shipped config carrying a removed field
    (such as ``enabled``, which this config no longer has) would load and quietly mean something else. Every field in
    every shipped exo config must therefore be one the config class knows.
    """
    from rl_train.envs.environment_handler import EnvironmentHandler
    from rl_train.train.train_configs.config_imitation import (
        ImitationTrainSessionConfig,
    )
    from rl_train.utils.data_types import DictionableDataclass

    shipped = sorted(CONFIG_DIR.parent.glob("exoboot_*/*.json"))
    assert SPLINE in shipped and EXO_OFF in shipped
    for path in shipped:
        saved = json.loads(path.read_text())["env_params"]
        config_type = EnvironmentHandler.get_config_type_from_session_id(saved["env_id"])
        restored = DictionableDataclass.to_dict(DictionableDataclass.create(config_type, {"env_params": saved}))
        restored = restored["env_params"]
        unknown = sorted(set(saved["exo_controller_params"]) - set(restored["exo_controller_params"]))
        assert not unknown, f"{path.name}: fields the config class does not have: {unknown}"
        assert {k: restored["exo_controller_params"][k] for k in saved["exo_controller_params"]} == saved[
            "exo_controller_params"
        ], path.name
        assert restored["device_controller"] == saved.get("device_controller", ""), path.name
    lossy = DictionableDataclass.create(ImitationTrainSessionConfig, json.loads(SPLINE.read_text()))
    assert not hasattr(lossy.env_params, "exo_controller_params")


# --- TickSchedule ---------------------------------------------------------------------------------------------------


def _tick_substeps(schedule, n_substeps):
    return [n for n in range(n_substeps) if schedule.tick()]


def test_150_hz_in_1200_hz_physics_ticks_every_8_substeps():
    """The 4PTS default: 1200 / 150 is a whole number, so every tick is exactly on time, 8 substeps apart."""
    ticks = _tick_substeps(TickSchedule(rate_hz=150, physics_rate_hz=1200), 1200 * 10)
    assert len(ticks) == 1500
    assert ticks == list(range(0, 1200 * 10, 8))


def test_175_hz_in_1200_hz_physics():
    """The boot's own rate. Ticks 7,7,7,7,7,7,6 substeps apart: exactly 175 per simulated second, never early, at most
    one substep late."""
    schedule = TickSchedule(rate_hz=175, physics_rate_hz=1200)
    ticks = _tick_substeps(schedule, 1200 * 10)
    assert len(ticks) == 1750
    assert ticks[:8] == [0, 7, 14, 21, 28, 35, 42, 48]
    assert set(np.diff(ticks)) == {6, 7}
    for k, substep in enumerate(ticks):
        lag = substep / 1200 - k / 175
        assert -1e-12 <= lag < 1 / 1200, f"tick {k} is {lag * 1e3:.3f} ms off its ideal time"


def test_schedule_restarts_on_reset():
    schedule = TickSchedule(rate_hz=175, physics_rate_hz=1200)
    first = _tick_substeps(schedule, 100)
    schedule.reset()
    assert _tick_substeps(schedule, 100) == first


def test_schedule_at_the_physics_rate_ticks_every_substep():
    assert _tick_substeps(TickSchedule(rate_hz=1200, physics_rate_hz=1200), 50) == list(range(50))


def test_schedule_faster_than_the_physics_is_refused():
    with pytest.raises(ValueError, match="physics rate"):
        TickSchedule(rate_hz=1201, physics_rate_hz=1200)


# --- The ExoBoot four-point spline (4PTS) at 150 Hz -----------------------------------------------------------------


def test_4pts_config_is_the_exo_off_experiment_with_the_controller():
    """The baseline and the 4PTS run must be the same experiment but for the controller: same rewards, policy and
    controller parameters, the exo-off run on the stock env and the 4PTS run on the device env."""
    off, on = (json.loads(p.read_text()) for p in (EXO_OFF, SPLINE))
    assert off["env_params"].pop("env_id") == "myoAssistLegImitationExo-v0"
    assert on["env_params"].pop("env_id") == DEVICE_ENV_ID
    assert on["env_params"].pop("device_controller") == "exoboot_spline"
    assert off["env_params"].pop("device_controller", "") == ""
    assert on["env_params"]["exo_controller_params"].pop("controller_rate_hz") == 150.0
    off["env_params"]["exo_controller_params"].pop("controller_rate_hz", None)
    assert off == on


@pytest.fixture(scope="module")
def spline_env():
    env = _make(_config(SPLINE))
    yield env
    env.close()


def test_4pts_runs_on_its_own_150_hz_schedule_and_holds_torque_in_between(spline_env, monkeypatch):
    """Each tick steps both legs with the sim time and that leg's foot GRF; torque is held on the substeps in between."""
    env = spline_env
    controller = env.device_controller
    assert [env.sim.model.actuator(i).name for i in controller.actuator_ids] == list(EXO_ACTUATORS)
    seen = {leg.side: [] for leg in controller.legs}
    for leg in controller.legs:

        def step(t, signal, side=leg.side):
            assert signal == env._get_foot_force(side), f"{side}: controller signal is not the env's foot force"
            seen[side].append(t)
            return 20.0 + len(seen[side])  # a new plantarflexion torque every tick

        monkeypatch.setattr(leg.controller, "step", step)
    env.reset()
    exo_ids = _exo_ids(env)
    felt = []
    for _ in range(7):
        env.step(env.action_space.sample())
        felt.append([_joint_torque(env, i) for i in exo_ids])
    dt = env.sim.model.opt.timestep
    expected_ticks = [s * dt for s in _tick_substeps(TickSchedule(rate_hz=150, physics_rate_hz=1200), 7 * 40)]
    for side, times in seen.items():
        assert times == pytest.approx(expected_ticks, abs=1e-12), side
    # The last substep of each control step is held from the latest tick before it.
    for step_index, torques in enumerate(felt):
        last_substep = 40 * step_index + 39
        n_ticks = sum(1 for t in expected_ticks if t <= last_substep * dt + 1e-12)
        assert torques == pytest.approx([-(20.0 + n_ticks)] * 2, abs=1e-9), f"step {step_index}"


def test_4pts_config_trains_in_subprocess_envs():
    """Past the first PPO update with the envs built in SubprocVecEnv workers, as training builds them."""
    from tools.tests.conftest import build_env, take_a_few_ppo_steps

    config, env = build_env(SPLINE, num_envs=2)
    try:
        assert type(env).__name__ == "SubprocVecEnv"
        take_a_few_ppo_steps(config, env)
    finally:
        env.close()


def test_4pts_exo_phase_valid_reports_the_in_loop_legs(spline_env, monkeypatch):
    env = spline_env
    for leg in env.device_controller.legs:
        monkeypatch.setattr(leg.controller, "diagnostics", lambda: {"phase_valid": 1.0})
    env.reset()
    *_, info = env.step(env.action_space.sample())
    assert info["rwd_dict"]["exo_phase_valid"] == 1.0


def test_the_diagnostic_does_not_change_the_reward(spline_env, monkeypatch):
    """exo_phase_valid has weight 0 in the configs: whatever it reads, the dense reward must not move."""
    env = spline_env
    assert env.rwd_keys_wt["exo_phase_valid"] == 0.0
    env.reset()
    env.step(env.action_space.sample())
    dense = {}
    for valid in (0.0, 1.0):
        for leg in env.device_controller.legs:
            monkeypatch.setattr(leg.controller, "diagnostics", lambda v=valid: {"phase_valid": v})
        rwd_dict = env.get_reward_dict(env.obs_dict)
        assert rwd_dict["exo_phase_valid"] == valid
        dense[valid] = np.asarray(rwd_dict["dense"]).copy()
    assert np.array_equal(dense[0.0], dense[1.0])
