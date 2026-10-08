"""The ExoBoot DL task as an in-loop controller: its parts against the boot's logic, and in the env.

The boot-side logic (events from is_stance edges, the state machine, the spline in stance) was also replayed against the
DL validation sessions' logs with tools/replay_dl_session.py; these pin the port's behaviour down where no log is needed.
"""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

from myoassist_utils.exo_ctrl import BootStateMachine, FourPointSpline, ShadowDevice, SpeedActivation, TickSchedule
from myoassist_utils.exo_ctrl.boot_state import REEL_IN, STANCE
from myoassist_utils.exo_ctrl.dl_controller import NET_HEADS, DLLeg, DLLegDrive, ExoBootDLDevice
from myoassist_utils.exo_ctrl.gait_net import StreamingGaitNet
from tools.tests.conftest import REPO_ROOT
from tools.tests.test_gait_net import random_weights, walking_samples

SPLINE = dict(rise_fraction=0.278, peak_fraction=0.543, fall_fraction=0.641, peak_torque=25.0, bias_torque=3.0)
DL_DIR = REPO_ROOT / "rl_train/train/train_configs/exoboot_dl"
DL = DL_DIR / "imitation_22_DephyExoBoot_L1_exoboot_dl.json"
DL_SHADOW = DL_DIR / "imitation_22_DephyExoBoot_L1_exoboot_dl_shadow.json"
SPLINE_CONFIG = REPO_ROOT / "rl_train/train/train_configs/exoboot_spline/imitation_22_DephyExoBoot_L1_exoboot_spline.json"


# --- speed activation ------------------------------------------------------------------------------------------------


def test_speed_activation_has_the_boots_hysteresis():
    speed = SpeedActivation(on_speed=0.7, off_speed=0.5)
    swing_only = True
    trace = []
    for v in [0.2, 0.69, 0.7, 0.9, 0.6, 0.51, 0.5, 0.45, 0.65, 0.71]:
        swing_only = speed.update(v, swing_only)
        trace.append(swing_only)
    #          0.2   0.69  0.7    0.9    0.6    0.51   0.5   0.45  0.65  0.71
    assert trace == [True, True, False, False, False, False, True, True, True, False]


def test_a_first_speed_above_threshold_turns_assist_on():
    """The boot's queue starts at 0, so a first reading at or above 0.7 counts as crossing up."""
    assert SpeedActivation().update(1.2, True) is False


def _machine():
    return BootStateMachine(reel_in_time=0.15, reel_out_time=0.17)


# --- one leg ---------------------------------------------------------------------------------------------------------


def _leg(assist_on="always"):
    return DLLeg(spline=FourPointSpline(**SPLINE), state_machine=_machine(), assist_on=assist_on, speed=SpeedActivation())


def test_events_are_the_edges_of_is_stance_and_phase_is_scaled():
    leg = _leg()
    assert leg.step(0.0, None) == 0.0 and leg.gait_phase is None, "nothing changes before the first reply"
    leg.step(0.01, (0.5, 0.0, 1.2))
    assert not leg.did_heel_strike
    leg.step(0.02, (1.4, 1.0, 1.2))
    assert leg.did_heel_strike and leg.gait_phase == pytest.approx(0.6), "phase clipped to 1, then x 0.6"
    leg.step(0.03, (0.2, 1.0, 1.2))
    assert not leg.did_heel_strike and leg.gait_phase == pytest.approx(0.12)
    leg.step(0.04, (0.9, 0.0, 1.2))
    assert leg.did_toe_off


def test_torque_is_the_spline_in_stance_only():
    leg = _leg()
    spline = FourPointSpline(**SPLINE)
    t, torques, states = 0.0, [], []
    for k in range(200):  # 1 s of ticks: stance from the heel strike at 0.25 s
        is_stance = 1.0 if 0.25 <= t < 0.85 else 0.0
        phase = max(0.0, t - 0.25) / 0.6
        torques.append(leg.step(t, (phase, is_stance, 1.2)))
        states.append(leg.state)
        t += 0.005
    for torque, state, k in zip(torques, states, range(200)):
        if state == STANCE:
            assert torque == pytest.approx(spline.torque(0.6 * min(1.0, max(0.0, k * 0.005 - 0.25) / 0.6)))
        else:
            assert torque == 0.0
    assert STANCE in states and REEL_IN in states


def test_speed_mode_starts_unassisted():
    leg = _leg("speed")
    leg.step(0.0, (0.5, 1.0, 0.0))
    assert leg.swing_only and leg.diagnostics()["assist_on"] == 0.0


# --- the device ------------------------------------------------------------------------------------------------------


class _Front:
    def __init__(self, samples):
        self.samples, self.i = samples, 0

    def reset(self):
        self.i = 0

    def sample(self, sim):
        self.i += 1
        return list(self.samples[self.i - 1])


def _device(samples_by_leg, *, rate=100.0, physics=100.0, weights=None):
    legs = [
        DLLegDrive(side=side, actuator_id=i, controller=_leg(), front_end=_Front(samples))
        for i, (side, samples) in enumerate(zip("rl", samples_by_leg))
    ]
    return ExoBootDLDevice(
        legs,
        weights=weights or random_weights(),
        schedule=TickSchedule(rate_hz=rate, physics_rate_hz=physics),
        dtype=np.float64,
    )


def _sim():
    return types.SimpleNamespace(data=types.SimpleNamespace(time=0.0))


def test_each_leg_uses_the_reply_to_the_previous_tick_with_the_jetsons_rounding(monkeypatch):
    weights = random_weights()
    samples = [walking_samples(30, seed=s) + 1e-7 for s in (1, 2)]  # extra digits the Pi's '%.5f' drops
    device = _device(samples, weights=weights)
    received = {0: [], 1: []}
    for i, leg in enumerate(device.legs):
        monkeypatch.setattr(leg.controller, "step", lambda t, reply, i=i: received[i].append(reply) or 0.0)
    sim = _sim()
    for _ in range(30):
        device.compute_torque(sim)
    ref = StreamingGaitNet(weights, n_streams=2, heads=NET_HEADS)
    expected = [None]
    for k in range(29):
        out = ref.step(np.round(np.stack([samples[0][k], samples[1][k]]), 5))
        expected.append(out)
    for i in range(2):
        assert received[i][0] is None
        for k in range(1, 30):
            sp, ss, v = received[i][k]
            assert sp == round(float(expected[k]["stance_phase"][i]), 5)
            assert ss == float(np.round(expected[k]["stance_swing"][i]))
            assert v == round(float(expected[k]["velocity"][i]), 5)


def test_torque_is_held_between_ticks_and_flipped_to_the_joint(monkeypatch):
    device = _device([walking_samples(40, seed=1), walking_samples(40, seed=2)], rate=175.0, physics=1200.0)
    for leg in device.legs:
        monkeypatch.setattr(leg.controller, "step", lambda t, reply: 12.0)
    sim = _sim()
    first = device.compute_torque(sim)
    assert first == [-12.0, -12.0], "plantarflexion is negative ankle torque"
    for _ in range(6):  # substeps 1..6 are not ticks at 175 Hz in 1200 Hz physics
        assert device.compute_torque(sim) is first


def test_shadow_mode_applies_nothing(monkeypatch):
    device = ShadowDevice(_device([walking_samples(10, seed=1), walking_samples(10, seed=2)]))
    for leg in device.legs:
        monkeypatch.setattr(leg.controller, "step", lambda t, reply: 25.0)
    sim = _sim()
    for _ in range(10):
        assert list(device.compute_torque(sim)) == [0.0, 0.0]
    assert device.inner._torques == [-25.0, -25.0], "the DL ran and would have applied its torque"


def test_reset_starts_the_buffer_and_the_legs_over():
    samples = [walking_samples(20, seed=1), walking_samples(20, seed=2)]
    device = _device(samples)
    sim = _sim()
    for _ in range(10):
        device.compute_torque(sim)
    device.reset()
    assert all(leg.front_end.i == 0 for leg in device.legs)
    assert all(leg.controller.gait_phase is None for leg in device.legs)


# --- in the env ------------------------------------------------------------------------------------------------------


def test_dl_configs_are_the_4pts_experiment_with_the_dl_controller():
    """The 4PTS run with the DL controller instead, which also runs at its own 175 Hz rather than the 4PTS's 150."""
    spline, dl, shadow = (json.loads(p.read_text()) for p in (SPLINE_CONFIG, DL, DL_SHADOW))
    assert dl["env_params"].pop("device_controller") == "exoboot_dl"
    assert spline["env_params"].pop("device_controller") == "exoboot_spline"
    dl_params, spline_params = dl["env_params"].pop("exo_controller_params"), spline["env_params"].pop("exo_controller_params")
    assert (dl_params["controller_rate_hz"], spline_params.pop("controller_rate_hz")) == (175.0, 150.0)
    assert {k: v for k, v in dl_params.items() if k in spline_params} == spline_params
    assert dl == spline
    assert shadow["env_params"]["exo_controller_params"].pop("shadow_mode") is True
    assert dl_params.pop("shadow_mode") is False
    shadow_params = shadow["env_params"].pop("exo_controller_params")
    shadow["env_params"].pop("device_controller")
    assert shadow_params == dl_params and shadow == dl


@pytest.mark.parametrize("rate", [150.0, 200.0])
def test_the_dl_controller_refuses_a_rate_other_than_175_hz(rate):
    """Its network was trained on the boot's 175 Hz samples; at another rate its window and filters would be wrong."""
    from tools.tests.test_device_control import _config, _make

    config = _config(DL)
    config.env_params.exo_controller_params.controller_rate_hz = rate
    with pytest.raises(ValueError, match="175 Hz"):
        _make(config)


def test_the_dl_env_runs_and_reports_its_legs():
    from tools.tests.test_device_control import _config, _make

    env = _make(_config(DL))
    try:
        assert type(env.device_controller).__name__ == "ExoBootDLDevice"
        names = [env.sim.model.sensor(i).name for i in range(env.sim.model.nsensor)]
        assert {"boot_imu_accel_r", "boot_imu_gyro_r", "boot_imu_accel_l", "boot_imu_gyro_l"} <= set(names)
        env.reset()
        for _ in range(20):
            *_, info = env.step(env.action_space.sample())
            assert np.all(np.isfinite(env.sim.data.ctrl))
        assert 0.0 <= float(info["rwd_dict"]["exo_phase_valid"]) <= 1.0
    finally:
        env.close()


def test_dl_torque_reaches_the_ankles(monkeypatch):
    from tools.tests.test_device_control import _config, _joint_torque, _make

    env = _make(_config(DL))
    try:
        for leg in env.device_controller.legs:
            monkeypatch.setattr(leg.controller, "step", lambda t, reply: 18.0)
        env.reset()
        env.step(env.action_space.sample())
        for actuator in env.device_controller.actuator_ids:
            assert _joint_torque(env, actuator) == pytest.approx(-18.0, abs=1e-6)
    finally:
        env.close()


def test_config_paths_are_resolved_against_the_repo_root():
    from myoassist_utils.exo_ctrl.dl_controller import repo_path

    assert repo_path("rl_train/x.npz") == REPO_ROOT / "rl_train/x.npz"
    assert repo_path(str(REPO_ROOT / "y.npz")) == REPO_ROOT / "y.npz"


def _filter_file(tmp_path, sides="rl"):
    from myoassist_utils.exo_ctrl.boot_sensors import save_out_of_plane_filters
    from tools.tests.test_boot_sensors import synthetic_filter

    path = tmp_path / "out_of_plane.npz"
    save_out_of_plane_filters(path, {side: synthetic_filter(seed) for seed, side in enumerate(sides)})
    return path


def test_the_dl_env_synthesizes_the_out_of_plane_channels_with_a_filter(tmp_path):
    from tools.tests.test_device_control import _config, _make

    config = _config(DL)
    config.env_params.exo_controller_params.dl_out_of_plane_filter_path = str(_filter_file(tmp_path))
    env = _make(config)
    try:
        filters = [leg.front_end.out_of_plane for leg in env.device_controller.legs]
        assert all(f is not None for f in filters) and filters[0] is not filters[1]
        env.reset()
        for _ in range(10):
            env.step(env.action_space.sample())
            assert np.all(np.isfinite(env.sim.data.ctrl))
        assert all(f._count > 0 for f in filters), "the filters ran"
    finally:
        env.close()
    config.env_params.exo_controller_params.dl_out_of_plane_filter_path = str(_filter_file(tmp_path, sides="r"))
    with pytest.raises(ValueError, match="need 'r' and 'l'"):
        _make(config)


@pytest.mark.parametrize("with_filter", [False, True])
def test_the_shadow_dl_env_reproduces_the_stock_exo_off_env_exactly(tmp_path, with_filter):
    """IMUs in the model, the network and the state machine running every tick, no torque: physics must be untouched,
    with the planar channels or with synthesized ones."""
    from tools.tests.test_device_control import EXO_OFF, _config, _rollout

    n_steps = 60
    stock, _ = _rollout(_config(EXO_OFF, flag_random_ref_index=False), n_steps)
    config = _config(DL_SHADOW, flag_random_ref_index=False)
    if with_filter:
        config.env_params.exo_controller_params.dl_out_of_plane_filter_path = str(_filter_file(tmp_path))
    shadow, calls = _rollout(config, n_steps)
    assert calls == 40 * n_steps
    for k, (s, d) in enumerate(zip(stock, shadow, strict=True)):
        for key in ("qpos", "qvel", "act", "obs"):
            assert np.array_equal(s[key], d[key]), f"step {k}: {key}"
        assert (s["time"], s["reward"], s["done"]) == (d["time"], d["reward"], d["done"]), f"step {k}"
