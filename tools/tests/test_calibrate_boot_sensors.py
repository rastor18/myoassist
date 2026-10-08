"""tools/calibrate_boot_sensors.py on synthetic data: a session logged through an IMU with a known mount on the shank."""

from __future__ import annotations

import csv
import re
import types

import numpy as np
import pandas as pd
import pytest

from myoassist_utils.exo_ctrl.boot_sensors import FRAME_SIGNS, rotation_from_rpy_deg, rpy_deg_from_rotation
from myoassist_utils.exo_ctrl.gait_net import INPUT_CHANNELS
from tools.calibrate_boot_sensors import (
    OUT_OF_PLANE,
    OutOfPlaneFilter,
    fit_mount,
    kabsch,
    mean_rotation,
    network_effects,
    perturbations,
    site_vectors,
    standing,
    stride_profiles,
    swing_axis,
    with_vectors,
)
from tools.replay_dl_session import TRAILING
from tools.tests.test_gait_net import random_weights, walking_samples

MOUNT = {"l": (1.0, 11.0, -5.0), "r": (-2.0, -14.5, -3.5)}  # roll, pitch, yaw: about what the DL sessions' boots measure
STANDING_ANKLE = {"l": -9.3, "r": -2.0}


def test_rpy_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        rpy = (rng.uniform(-170, 170), rng.uniform(-80, 80), rng.uniform(-170, 170))
        assert np.allclose(rpy_deg_from_rotation(rotation_from_rpy_deg(*rpy)), rpy, atol=1e-9)


def test_kabsch_recovers_a_rotation_from_two_directions():
    r = rotation_from_rpy_deg(10.0, -20.0, 30.0)
    u = [np.array([0.0, 1.0, 0.0]), np.array([0.3, 0.1, 0.9]) / np.linalg.norm([0.3, 0.1, 0.9])]
    assert np.allclose(kabsch([(a, r @ a) for a in u]), r, atol=1e-12)


@pytest.mark.parametrize("side", ["l", "r"])
def test_fit_mount_recovers_a_known_mount(side):
    to_imu = rotation_from_rpy_deg(*MOUNT[side]).T
    gravity = to_imu @ np.array([0.0, 1.0, 0.0]) * 1.012  # accelerometer scale error does not matter
    axis = to_imu @ np.array([0.0, 0.0, 1.0])
    fitted, rpy = fit_mount(gravity, axis)
    assert np.allclose(fitted, to_imu, atol=1e-12)
    assert np.allclose(rpy, MOUNT[side], atol=1e-9)


def test_mean_rotation_of_nearby_rotations():
    rotations = [rotation_from_rpy_deg(1.0, 11.0 + d, -5.0) for d in (-0.5, 0.5)]
    assert np.allclose(rpy_deg_from_rotation(mean_rotation(rotations)), (1.0, 11.0, -5.0), atol=1e-2)


def test_the_fitted_mount_makes_the_keyframe_read_the_logged_gravity():
    """Through the env's own model and front end: the calibration closes the loop at the standing keyframe."""
    import mujoco

    from myoassist_utils.exo_ctrl import BootSensorFrontEnd
    from tools.calibrate_boot_sensors import build_model

    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)
    for side in "lr":
        to_imu = rotation_from_rpy_deg(*MOUNT[side]).T
        logged = (to_imu @ np.array([0.0, 1.0, 0.0])) * FRAME_SIGNS[side][0]  # in the boot's frame
        _, rpy = fit_mount(logged * FRAME_SIGNS[side][0], to_imu @ np.array([0.0, 0.0, 1.0]))
        front = BootSensorFrontEnd(model, side, standing_angle_deg=STANDING_ANKLE[side], mount_rpy_deg=rpy, quantize=False)
        reading = np.array(front.sample(types.SimpleNamespace(data=data)))
        assert np.allclose(reading[:3] / np.linalg.norm(reading[:3]), logged, atol=1e-6)
        assert reading[6] == pytest.approx(STANDING_ANKLE[side], abs=1e-9)


def _planar_vectors(n, rng):
    accel = np.column_stack([rng.normal(0, 0.5, n), 1.0 + rng.normal(0, 0.5, n), np.zeros(n)])
    gyro = np.column_stack([np.zeros(n), np.zeros(n), rng.normal(0, 150, n)])
    return accel, gyro


@pytest.mark.parametrize("side", ["l", "r"])
def test_mount_crosstalk_only_keeps_planar_motion_and_zeroing_zeroes(side):
    rng = np.random.default_rng(1)
    n = 500
    to_imu = rotation_from_rpy_deg(*MOUNT[side]).T
    accel, gyro = _planar_vectors(n, rng)
    x = with_vectors(rng.normal(size=(n, 8)), side, accel @ to_imu.T, gyro @ to_imu.T)  # what a mounted IMU logs
    variants = perturbations(x, side, to_imu, ankle_mean=0.0)
    assert np.allclose(variants["mount crosstalk only"], x, atol=1e-12)
    assert np.all(variants["out-of-plane zeroed"][:, [2, 3, 4]] == 0.0)
    assert np.array_equal(variants["out-of-plane zeroed"][:, [0, 1, 5, 6, 7]], x[:, [0, 1, 5, 6, 7]])


def test_site_vectors_undo_the_left_legs_handedness():
    x = np.arange(16, dtype=float).reshape(2, 8)
    for side in "lr":
        accel, gyro = site_vectors(x, side)
        assert np.array_equal(with_vectors(x, side, accel, gyro), x)
    accel, gyro = site_vectors(x, "l")
    assert np.array_equal(accel[:, 2], -x[:, 2]) and np.array_equal(gyro[:, :2], -x[:, 3:5])


def test_standing_rejects_a_window_with_motion():
    t = np.arange(0, 5, 1 / 175)
    x = np.zeros((len(t), 8))
    x[:, 1] = 1.0
    log = types.SimpleNamespace(t=t, sent=x)
    assert standing(log, "r", (0.5, 4.5)).gravity == pytest.approx([0.0, 1.0, 0.0])
    x[300, 5] = 40.0
    with pytest.raises(ValueError, match="not quiet"):
        standing(log, "r", (0.5, 4.5))


def test_swing_axis_finds_the_turning_axis():
    rng = np.random.default_rng(2)
    n = 4000
    axis = np.array([-0.19, 0.07, 0.98])
    axis /= np.linalg.norm(axis)
    rate = 300 * np.abs(np.sin(np.arange(n) / 30))
    x = np.zeros((n, 8))
    x[:, 3:6] = rate[:, None] * axis * np.array(FRAME_SIGNS["l"][1]) + rng.normal(0, 2, (n, 3))
    log = types.SimpleNamespace(sent=x, is_stance=np.zeros(n))
    found, shares = swing_axis(log, "l", np.ones(n, dtype=bool))
    assert np.allclose(found, axis, atol=2e-3) and shares[0] > 0.99


def test_stride_profiles_average_strides_of_one_shape():
    t = np.arange(0, 30, 1 / 175)
    durations = np.resize([1.0, 1.1, 1.2], 30)
    starts = np.r_[0.0, np.cumsum(durations)]
    starts = starts[starts < t[-1]]
    phase = np.interp(t, starts, np.arange(len(starts)))
    x = np.sin(2 * np.pi * phase)[:, None]
    strikes = np.zeros(len(t), dtype=bool)
    strikes[np.searchsorted(t, starts[:-1])] = True
    profiles = stride_profiles(t, x, strikes)
    assert profiles.shape[1:] == (101, 1)
    assert np.allclose(profiles.mean(axis=0)[:, 0], np.sin(2 * np.pi * np.linspace(0, 1, 101)), atol=0.02)


def test_out_of_plane_filter_recovers_a_causal_linear_system():
    rng = np.random.default_rng(3)
    n = 6000

    def smooth(k):
        return np.convolve(rng.normal(size=n + 50), np.ones(25) / 25, mode="same")[:n] * k

    x = np.zeros((n, 8))
    accel = np.column_stack([smooth(1.0), smooth(1.0), np.zeros(n)])
    gyro = np.column_stack([np.zeros(n), np.zeros(n), smooth(100.0)])
    x[:, 6], x[:, 7] = smooth(10.0), smooth(100.0)
    # out-of-plane = a causal mix of the sagittal channels now and 0.1 s ago (lag 18 = one of the filter's lags)
    lagged = np.vstack([np.zeros((18, 3)), np.column_stack([accel[:, 0], gyro[:, 2], x[:, 7]])[:-18]])
    accel[:, 2] = 0.3 * accel[:, 1] - 0.2 * lagged[:, 0]
    gyro[:, 0] = 0.25 * gyro[:, 2] + 0.1 * lagged[:, 2]
    gyro[:, 1] = -0.4 * lagged[:, 1] + 2.0 * x[:, 6]
    rows = np.zeros(n, dtype=bool)
    rows[100:4000] = True
    f = OutOfPlaneFilter.fit([(accel, gyro, x, rows)], features=OutOfPlaneFilter.ALL_FEATURES, ridge=1e-9)
    test = np.zeros(n, dtype=bool)
    test[4100:] = True
    truth = np.column_stack([accel[:, 2], gyro[:, 0], gyro[:, 1]])[test]
    predicted = f.predict(accel, gyro, x)[test]
    assert np.all(1 - np.var(truth - predicted, axis=0) / np.var(truth, axis=0) > 0.999)
    # The default reads no absolute angle, so the sim's different ankle offset cannot leak into the channels.
    default = OutOfPlaneFilter.fit([(accel, gyro, x, rows)])
    shifted = x.copy()
    shifted[:, 6] -= 8.0
    assert np.array_equal(default.predict(accel, gyro, x), default.predict(accel, gyro, shifted))


def test_network_effects_of_the_unchanged_inputs_are_nil():
    weights = random_weights(seed=4)
    x = np.round(walking_samples(700, seed=5), 5)
    t = np.arange(len(x)) / 175.0
    rows = t > 1.0
    effects = network_effects(t, {"unchanged": x, "copy": x.copy(), "zeroed": np.zeros_like(x)}, weights, rows)
    copy = effects["copy"]
    assert copy["phase_rmse"] == 0.0 and copy["is_stance_agreement"] == 1.0 and copy["speed_bias"] == 0.0
    assert effects["zeroed"]["phase_rmse"] > 0.0


# --- end to end ----------------------------------------------------------------------------------------------------


def _write_session(prefix, rng):
    """5 s of quiet standing, then 35 s of a planar gait seen through each leg's MOUNT, with the network's is_stance."""
    t = np.arange(0, 40, 1 / 175)
    walking = t >= 5.0
    phase = ((t - 5.0) / 1.1) % 1.0
    stance = phase < 0.6
    swing = np.clip((phase - 0.6) / 0.4, 0, 1)
    rate = np.where(walking, np.where(stance, -60 * np.sin(np.pi * phase / 0.6), 380 * np.sin(np.pi * swing)), 0.0)
    tilt = np.where(walking, 0.4 * np.sin(2 * np.pi * phase), 0.0)
    pd.DataFrame([dict(loop_time=0.0, SWING_ONLY=False, LEFT_STANDING_ANGLE=-6.4, RIGHT_STANDING_ANGLE=3.7)]).to_csv(
        prefix + "CONFIG.csv", index=False
    )
    for side_name, side in (("LEFT", "l"), ("RIGHT", "r")):
        to_imu = rotation_from_rpy_deg(*MOUNT[side]).T
        accel = np.column_stack([-np.sin(tilt), np.cos(tilt), np.zeros(len(t))]) + rng.normal(0, 1e-3, (len(t), 3))
        gyro = np.column_stack([np.zeros(len(t)), np.zeros(len(t)), rate]) + rng.normal(0, 0.3, (len(t), 3))
        x = np.zeros((len(t), 8))
        x = with_vectors(x, side, accel @ to_imu.T, gyro @ to_imu.T)
        x[:, 6] = STANDING_ANKLE[side] + np.where(walking, 12 * np.sin(2 * np.pi * phase), 0.0)
        x[:, 7] = np.gradient(x[:, 6], t)
        with open(prefix + f"{side_name}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "state_time",
                    "loop_time",
                    *INPUT_CHANNELS,
                    "did_heel_strike",
                    "gait_phase",
                    "did_toe_off",
                    "commanded_torque",
                    "controller",
                    "control_state",
                ]
            )
            for i in range(len(t)):
                trailing = [0.6 * phase[i], float(stance[i]), 1.2, 0.0, 1.2, 0.0] if walking[i] else []
                writer.writerow([i, t[i], *np.round(x[i], 5), 0, "", 0, 0.0, 2, 2, *trailing])
    assert len(TRAILING) == 6


def test_the_front_ends_synthesis_is_the_tools_offline_synthesis_on_a_kinematic_replay():
    """The reference gait replayed through the env's model: each leg's front end with a filter, sample by sample, gives
    what the tool's offline synthesis makes of the same front end's planar channels -- the channels the filter was fit
    against are the channels it is run on, in the same frame."""
    from myoassist_utils.exo_ctrl import BootSensorFrontEnd
    from tools.calibrate_boot_sensors import build_model, replay_reference, synthesize
    from tools.tests.test_boot_sensors import synthetic_filter

    model = build_model()
    for seed, side in enumerate("rl"):
        kwargs = dict(standing_angle_deg=STANDING_ANKLE[side], mount_rpy_deg=MOUNT[side], quantize=False)
        f = synthetic_filter(seed)
        _, plain = replay_reference(model, {side: BootSensorFrontEnd(model, side, **kwargs)}, seconds=4.0, decimals=None)
        _, streamed = replay_reference(
            model, {side: BootSensorFrontEnd(model, side, out_of_plane=f, **kwargs)}, seconds=4.0, decimals=None
        )
        offline = synthesize(plain[side], side, rotation_from_rpy_deg(*MOUNT[side]).T, f)
        np.testing.assert_allclose(streamed[side], offline, rtol=1e-10, atol=1e-10)
        assert np.abs(streamed[side][:, OUT_OF_PLANE] - plain[side][:, OUT_OF_PLANE]).max() > 1.0, "the filter did act"


def test_save_filter_fits_each_leg_and_reports_each_session_left_out(tmp_path, capsys):
    from myoassist_utils.exo_ctrl.boot_sensors import load_out_of_plane_filters
    from myoassist_utils.exo_ctrl.gait_net import save_weights
    from tools.calibrate_boot_sensors import main

    prefixes = [str(tmp_path / f"s{k}_") for k in range(2)]
    for k, prefix in enumerate(prefixes):
        _write_session(prefix, np.random.default_rng(10 + k))
    weights = tmp_path / "w.npz"
    save_weights(weights, random_weights(seed=4))
    out = tmp_path / "filter.npz"
    main([*prefixes, "--walk-start", "8", "--seconds", "8", "--weights", str(weights), "--save-filter", str(out)])
    filters = load_out_of_plane_filters(out)
    assert set(filters) == {"r", "l"}
    for f in filters.values():
        assert f.features == OutOfPlaneFilter.FEATURES and f.rate_hz == 175.0 and f.ridge == 1e-3
        assert len(f.loo_rmse) == 2 and all(np.isfinite(f.loo_rmse))
    assert "left out of the fit" in capsys.readouterr().out


def test_save_filter_needs_the_weights(tmp_path):
    from tools.calibrate_boot_sensors import main

    with pytest.raises(SystemExit):
        main([str(tmp_path / "s_"), "--save-filter", str(tmp_path / "f.npz")])


def test_main_recovers_the_mount_and_the_standing_angle(tmp_path, capsys):
    from tools.calibrate_boot_sensors import main

    prefix = str(tmp_path / "20990101_0000_TEST_")
    _write_session(prefix, np.random.default_rng(6))
    calibration = main([prefix, "--walk-start", "8", "--seconds", "8"])
    out = capsys.readouterr().out
    for side in "lr":
        assert calibration[side]["standing_angle_deg"] == pytest.approx(STANDING_ANKLE[side], abs=0.01)
        assert np.allclose(calibration[side]["mount_rpy_deg"], MOUNT[side], atol=0.3)
    assert "1. standing" in out and "2. mount" in out and "3. profiles" in out and "4. network" not in out
    assert re.search(r'"imu_mount_l_pitch_deg": 1[01]\.\d+', out)
