"""The simulated boot sensors: an IMU in each actuator pack, read the way the ExoBoot reads its own.

What has to hold: adding the IMUs changes nothing else in the model; the readings are physics (gravity at rest, the
finite-differenced acceleration in motion); and the conversion gives the boot's units and signs on both legs, the left
leg's frame being left-handed.
"""

from __future__ import annotations

import json
import math
import types

import mujoco
import numpy as np
import pytest

from myoassist_utils.compose import compose_env_model
from myoassist_utils.exo_ctrl import BootSensorFrontEnd, add_boot_imus
from myoassist_utils.exo_ctrl.boot_sensors import (
    ACCEL,
    ACCEL_LSB_G,
    FRAME_SIGNS,
    GYRO,
    GYRO_LSB_DEG_S,
    OUT_OF_PLANE_FORMAT,
    OutOfPlaneFilter,
    load_out_of_plane_filters,
    rotation_from_rpy_deg,
    save_out_of_plane_filters,
)


@pytest.fixture(scope="module")
def xmls():
    plain = compose_env_model("myolegs22", "DephyExoBoot_L1")
    return plain, add_boot_imus(plain)


@pytest.fixture(scope="module")
def model(xmls):
    return mujoco.MjModel.from_xml_string(xmls[1])


def _at_rest(model, *, knee=0.0):
    """Keyframe pose, knees at ``knee``, zero velocity and acceleration, sensors from inverse dynamics."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for side in "rl":
        data.qpos[model.jnt_qposadr[model.joint(f"knee_angle_{side}").id]] = knee
    data.qvel[:] = 0
    data.qacc[:] = 0
    mujoco.mj_inverse(model, data)
    return data


def _front(model, side, **kwargs):
    return BootSensorFrontEnd(model, side, standing_angle_deg=kwargs.pop("standing_angle_deg", 0.0), **kwargs)


def test_imus_change_nothing_else(xmls):
    """Same dynamics bit for bit, and every existing sensor keeps its name and address."""
    plain, with_imus = (mujoco.MjModel.from_xml_string(x) for x in xmls)
    assert [plain.sensor(i).name for i in range(plain.nsensor)] == [with_imus.sensor(i).name for i in range(plain.nsensor)]
    assert np.array_equal(plain.sensor_adr, with_imus.sensor_adr[: plain.nsensor])
    ctrls = np.random.default_rng(0).uniform(0, 1, size=(300, plain.nu))
    final = []
    for m in (plain, with_imus):
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        for c in ctrls:
            d.ctrl[:] = c
            mujoco.mj_step(m, d)
        final.append((d.qpos.copy(), d.qvel.copy(), d.act.copy()))
    for a, b in zip(*final):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("knee", [0.0, -0.3, -0.8])
def test_a_shank_at_rest_reads_gravity(model, knee):
    """(-sin theta, cos theta, 0) g on both legs, theta the forward lean of the shank."""
    data = _at_rest(model, knee=knee)
    up = data.joint("knee_angle_r").xanchor - data.joint("ankle_angle_r").xanchor
    theta = math.atan2(up[0], up[2])
    for side in "rl":
        accel = _front(model, side, quantize=False).sample(types.SimpleNamespace(data=data))[:3]
        assert accel == pytest.approx([-math.sin(theta), math.cos(theta), 0.0], abs=1e-9), side


def test_forward_swing_reads_positive_gyro_z_on_both_legs(model):
    """Knee extension swings the foot forward, which the boot's heel-strike detector expects as positive gyro_z."""
    for side in "rl":
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qvel[:] = 0
        data.qvel[model.jnt_dofadr[model.joint(f"knee_angle_{side}").id]] = 1.0
        mujoco.mj_forward(model, data)
        gyro = _front(model, side, quantize=False).sample(types.SimpleNamespace(data=data))[3:6]
        assert gyro == pytest.approx([0.0, 0.0, math.degrees(1.0)], abs=1e-9), side


def test_the_accelerometer_is_the_finite_differenced_acceleration(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    rng = np.random.default_rng(1)
    site = model.site("boot_imu_r").id
    adr = model.sensor_adr[model.sensor(ACCEL.format(side="r")).id]
    velocities, readings, rotations = [], [], []
    for k in range(400):
        if k % 40 == 0:
            data.ctrl[:] = rng.uniform(0, 1, model.nu)
        mujoco.mj_forward(model, data)
        v = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site, v, 0)
        velocities.append(v[3:].copy())
        readings.append(data.sensordata[adr : adr + 3].copy())
        rotations.append(data.site_xmat[site].reshape(3, 3).copy())
        mujoco.mj_step(model, data)
    acc = np.diff(velocities, axis=0) / model.opt.timestep
    expected = np.array([rotations[k].T @ (acc[k] - model.opt.gravity) for k in range(len(acc))])
    measured = np.array(readings[:-1])
    relative = np.linalg.norm(expected - measured, axis=1) / np.linalg.norm(measured, axis=1)
    assert np.median(relative) < 0.02


def test_units_signs_and_the_left_legs_mirrored_frame(model):
    """Right leg as the site frame; left leg with the lateral accel axis and the gyro's x and y components reversed."""
    data = mujoco.MjData(model)
    g = float(np.linalg.norm(model.opt.gravity))
    for side in "rl":
        a = model.sensor_adr[model.sensor(ACCEL.format(side=side)).id]
        w = model.sensor_adr[model.sensor(GYRO.format(side=side)).id]
        data.sensordata[a : a + 3] = [g * 0.1, g * 0.2, g * 0.3]
        data.sensordata[w : w + 3] = [0.1, 0.2, 0.3]
    sim = types.SimpleNamespace(data=data)
    right = _front(model, "r", quantize=False).sample(sim)
    left = _front(model, "l", quantize=False).sample(sim)
    deg = math.degrees
    assert right[:6] == pytest.approx([0.1, 0.2, 0.3, deg(0.1), deg(0.2), deg(0.3)])
    assert left[:6] == pytest.approx([0.1, 0.2, -0.3, -deg(0.1), -deg(0.2), deg(0.3)])


def test_quantization_is_the_dephys_resolution(model):
    data = _at_rest(model, knee=-0.37)
    sample = _front(model, "r").sample(types.SimpleNamespace(data=data))
    for value, lsb in zip(sample[:6], [ACCEL_LSB_G] * 3 + [GYRO_LSB_DEG_S] * 3):
        assert value / lsb == pytest.approx(round(value / lsb), abs=1e-6)


def test_ankle_angle_is_plantarflexion_positive_from_the_standing_angle(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    sim = types.SimpleNamespace(data=data)
    front = _front(model, "l", standing_angle_deg=-8.0, quantize=False)
    assert front.sample(sim)[6] == pytest.approx(-8.0, abs=1e-9), "the standing keyframe reads the standing angle"
    data.qpos[model.jnt_qposadr[model.joint("ankle_angle_l").id]] += 0.1  # dorsiflex by 0.1 rad
    assert front.sample(sim)[6] == pytest.approx(-8.0 - math.degrees(0.1), abs=1e-9)


def test_ankle_velocity_is_zero_on_the_first_read_then_filtered(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    sim = types.SimpleNamespace(data=data)
    front = _front(model, "r", quantize=False)
    adr = model.jnt_qposadr[model.joint("ankle_angle_r").id]
    assert front.sample(sim)[7] == 0.0
    data.time += 1 / 175
    data.qpos[adr] -= math.radians(1.0)  # 1 deg of plantarflexion in one tick
    # The filter starts from its first input, so the first filtered value is the raw difference quotient.
    assert front.sample(sim)[7] == pytest.approx(175.0, rel=1e-9)
    front.reset()
    assert front.sample(sim)[7] == 0.0


def test_a_tilted_mount_rotates_the_readings(model):
    data = _at_rest(model)
    sim = types.SimpleNamespace(data=data)
    tilted = _front(model, "r", quantize=False, mount_rpy_deg=(0.0, 0.0, 90.0)).sample(sim)
    assert tilted[:3] == pytest.approx([1.0, 0.0, 0.0], abs=1e-9), "yawing the IMU 90 deg puts gravity on its x axis"


def test_a_model_without_the_imus_is_refused(xmls):
    with pytest.raises(KeyError, match="add_boot_imus"):
        _front(mujoco.MjModel.from_xml_string(xmls[0]), "r")


# --- the out-of-plane filter -----------------------------------------------------------------------------------------


def synthetic_filter(seed=0, features=OutOfPlaneFilter.FEATURES, rate_hz=175.0):
    """A filter with random coefficients of the shape a fit gives: no boot log involved."""
    rng = np.random.default_rng(seed)
    n = 1 + len(OutOfPlaneFilter.LAGS) * len(features)
    return OutOfPlaneFilter(rng.normal(0, 0.01, (n, 3)), 0.0, features, rate_hz=rate_hz, ridge=1e-3, loo_rmse=(0.02, 0.03))


def _signals(n, seed):
    rng = np.random.default_rng(seed)
    accel = rng.normal(0, 0.5, (n, 3))
    gyro = rng.normal(0, 100, (n, 3))
    x = np.zeros((n, 8))
    x[:, 6], x[:, 7] = rng.normal(-5, 10, n), rng.normal(0, 100, n)
    return accel, gyro, x


@pytest.mark.parametrize("features", [OutOfPlaneFilter.FEATURES, OutOfPlaneFilter.ALL_FEATURES])
def test_the_streaming_filter_equals_the_offline_one(features):
    """Sample by sample from a reset, as the front end runs it, against the whole recording at once; the time before the
    first sample counts as that sample in both. Twice, so reset() must start it over."""
    f = synthetic_filter(features=features)
    f.ankle_mean = -3.0
    accel, gyro, x = _signals(400, seed=1)
    offline = f.predict(accel, gyro, x)
    for _ in range(2):
        f.reset()
        streamed = np.array([f.step(accel[i], gyro[i], x[i, 6], x[i, 7]) for i in range(len(x))])
        np.testing.assert_allclose(streamed, offline, rtol=1e-12, atol=1e-12)


def test_a_saved_filter_round_trips(tmp_path):
    filters = {"r": synthetic_filter(1), "l": synthetic_filter(2)}
    path = tmp_path / "f.npz"
    save_out_of_plane_filters(path, filters)
    loaded = load_out_of_plane_filters(path)
    assert set(loaded) == {"r", "l"}
    for side, f in filters.items():
        g = loaded[side]
        assert np.array_equal(g.coefficients, f.coefficients) and np.array_equal(g.lags, f.lags)
        assert (g.features, g.rate_hz, g.ridge, g.loo_rmse) == (f.features, 175.0, 1e-3, (0.02, 0.03))
    assert loaded["r"] is not loaded["l"], "each leg streams through its own"


def _write(path, metadata=None, **arrays):
    np.savez(path, **({} if metadata is None else {"metadata": np.array(json.dumps(metadata))}), **arrays)


def test_malformed_filter_files_are_refused(tmp_path):
    good = synthetic_filter()
    meta = {"format": OUT_OF_PLANE_FORMAT, "sides": {"r": good._metadata()}}
    arrays = {"r/coefficients": good.coefficients, "r/lags": good.lags}
    cases = {
        "no metadata": (None, arrays),
        "not an out-of-plane": ({"format": "something else"}, arrays),
        "need filters by side": ({**meta, "sides": {"x": good._metadata()}}, arrays),
        "lacks": (meta, {"r/coefficients": good.coefficients}),
        "coefficients have shape": (meta, {**arrays, "r/coefficients": good.coefficients[:-1]}),
        "not finite": (meta, {**arrays, "r/coefficients": np.full_like(good.coefficients, np.nan)}),
        "lags must increase": (meta, {**arrays, "r/lags": good.lags[::-1].copy()}),
        "distinct names": ({**meta, "sides": {"r": {**good._metadata(), "features": ["gyro_q", "ankle_velocity"]}}}, arrays),
    }
    for i, (message, (metadata, contents)) in enumerate(cases.items()):
        path = tmp_path / f"bad{i}.npz"
        _write(path, metadata, **contents)
        with pytest.raises(ValueError, match=message):
            load_out_of_plane_filters(path)
    weights = tmp_path / "weights.npz"
    _write(weights, {"engine": "not a filter"}, x=np.zeros(3))
    with pytest.raises(ValueError, match="not an out-of-plane"):
        load_out_of_plane_filters(weights)


def test_the_front_end_refuses_a_filter_fit_at_another_rate_or_on_the_ankle_angle(model):
    with pytest.raises(ValueError, match="fit at 200 Hz"):
        _front(model, "r", out_of_plane=synthetic_filter(rate_hz=200.0))
    with pytest.raises(ValueError, match="ankle offset"):
        _front(model, "r", out_of_plane=synthetic_filter(features=OutOfPlaneFilter.ALL_FEATURES))


def test_the_filter_replaces_the_out_of_plane_channels_before_the_mount_and_quantization(model):
    """At rest, gyro and ankle velocity 0: the synthesized site-frame channels are the intercept, then mounted, signed and
    quantized like any reading. The in-plane channels are the planar model's."""
    data = _at_rest(model, knee=-0.3)
    sim = types.SimpleNamespace(data=data)
    for side in "rl":
        f = synthetic_filter(3)
        intercept = f.coefficients[0]
        mount = (2.0, -14.0, -3.0)
        plain = BootSensorFrontEnd(model, side, standing_angle_deg=0.0, quantize=False).sample(sim)
        site = np.array(plain[:6])  # no mount: the site frame, signed
        site[[2, 3, 4]] = np.array(intercept) * np.array([FRAME_SIGNS[side][0][2], *FRAME_SIGNS[side][1][:2]])
        to_imu = rotation_from_rpy_deg(*mount).T
        accel_sign, gyro_sign = (np.array(s) for s in FRAME_SIGNS[side])
        expected = np.r_[accel_sign * (to_imu @ (accel_sign * site[:3])), gyro_sign * (to_imu @ (gyro_sign * site[3:6]))]
        front = BootSensorFrontEnd(model, side, standing_angle_deg=0.0, mount_rpy_deg=mount, out_of_plane=f)
        reading = front.sample(sim)
        np.testing.assert_allclose(reading[:6], expected, atol=GYRO_LSB_DEG_S)
        for value, lsb in zip(reading[:6], [ACCEL_LSB_G] * 3 + [GYRO_LSB_DEG_S] * 3):
            assert value / lsb == pytest.approx(round(value / lsb), abs=1e-6)
        assert reading[6:] == pytest.approx(plain[6:], abs=360 / 2**14), "the ankle, to one encoder click"


def test_the_front_ends_reset_clears_the_filters_history(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    sim = types.SimpleNamespace(data=data)
    front = _front(model, "r", quantize=False, out_of_plane=synthetic_filter(4))
    knee = model.jnt_dofadr[model.joint("knee_angle_r").id]

    def run():
        out = []
        for k in range(30):
            data.qvel[knee] = np.sin(k / 5.0)
            data.time = k / 175
            mujoco.mj_forward(model, data)
            out.append(front.sample(sim))
        return np.array(out)

    first = run()
    front.reset()
    assert np.array_equal(run(), first)
