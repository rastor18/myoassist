"""Phase 5: the simulated boot sensors against DL sessions' logs, and what the remaining differences do to the network.

Each leg of each session gets:

  1. standing  the quiet standing at the start of the log (``--standing``): gravity in the IMU frame, and the ankle angle
               as the boot read it. The logged *_STANDING_ANGLE is read during the boot's calibration with the cable
               taut, a few degrees off relaxed standing, so the sim's ankle offset comes from here instead.
  2. mount     the IMU's rotation on the shank, fit to two directions at once (Kabsch): standing gravity, against the
               keyframe, where the simulated shank is vertical; and the principal axis of the swing gyro, against the
               planar model's z. A shank that stands tilted reads as mount tilt: the two cannot be told apart here.
  3. profiles  stride-averaged channels between the ported gyro detector's heel strikes: the reference gait replayed
               kinematically through the calibrated sensor model, against the log -- r, peak-to-peak ratio, mean
               offset, peak timing. The two are different gaits (the reference is not the subject in the boot), so
               these show where the sim's inputs differ, not errors to fit away.
  4. network   the recovered network on the logged inputs with one sim-vs-boot difference applied at a time, against
               the unchanged inputs: stance-phase RMSE, is_stance agreement, heel-strike and toe-off shifts, the
               filtered speed. "out-of-plane zeroed" is the planar model without a mount; "mount crosstalk only" is
               what it gives with the fitted mount (decision gate 2). ``--synthesize`` adds out-of-plane channels
               predicted from the sagittal ones by a causal linear filter (``boot_sensors.OutOfPlaneFilter``) fit on
               the other sessions.

Prints the calibrated ExoControllerParams values, averaged over the sessions given. ``--save-filter <path>.npz`` fits
the out-of-plane filter per leg on every session's walking rows and saves it for ``dl_out_of_plane_filter_path``, with
the network's stance-phase RMSE on each session left out of the fit (a filter fit on the others):

    python tools/calibrate_boot_sensors.py "<dir>/<date>_<time>_<subject>_<trial>_" "<dir>/<...>_" \\
        --weights rl_train/train/train_configs/exoboot_dl/gait_net_4headed.npz --synthesize --save-filter <path>.npz

The filter's coefficients are fit to the sessions' logs; keep the file with them, outside the repo, unless their use
is cleared.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import types

import numpy as np
import pandas as pd

from myoassist_utils.exo_ctrl import Butterworth
from myoassist_utils.exo_ctrl.boot_sensors import (
    FRAME_SIGNS,
    BootSensorFrontEnd,
    OutOfPlaneFilter,
    add_boot_imus,
    rpy_deg_from_rotation,
    save_out_of_plane_filters,
)
from myoassist_utils.exo_ctrl.gait_net import INPUT_CHANNELS, StreamingGaitNet, load_weights

try:
    from tools.replay_dl_session import DlLog
    from tools.replay_exoboot_log import GYRO_PARAMS, replay_gyro_strikes
except ModuleNotFoundError:  # run as a script, with tools/ itself on sys.path
    from replay_dl_session import DlLog
    from replay_exoboot_log import GYRO_PARAMS, replay_gyro_strikes

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DL_CONFIG = REPO_ROOT / "rl_train/train/train_configs/exoboot_dl/imitation_22_DephyExoBoot_L1_exoboot_dl.json"
REFERENCE = REPO_ROOT / "rl_train/reference_data/short_reference_gait.npz"
SIDES = {"LEFT": "l", "RIGHT": "r"}
RATE = 175.0
GYRO = {arg: default for arg, default in GYRO_PARAMS.values()}  # the boot's default gyro detector
HEADS = ("stance_swing", "stance_phase", "velocity")  # what the boot's control uses
SPEED_FILTER = dict(order=2, cutoff_hz=0.5, fs_hz=200.0)
OUT_OF_PLANE = (2, 3, 4)  # accel_z, gyro_x, gyro_y in INPUT_CHANNELS


# --- frames ------------------------------------------------------------------------------------------------------------


def site_vectors(x: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """The boot's accel and gyro ([n, 3] each) with the left leg's left-handed frame made right-handed."""
    accel_sign, gyro_sign = FRAME_SIGNS[side]
    return x[:, 0:3] * accel_sign, x[:, 3:6] * gyro_sign


def with_vectors(x: np.ndarray, side: str, accel: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """A copy of the boot's 8 channels ``x`` with accel and gyro replaced by right-handed ones, put back in its frame."""
    accel_sign, gyro_sign = FRAME_SIGNS[side]
    y = x.copy()
    y[:, 0:3] = accel * accel_sign
    y[:, 3:6] = gyro * gyro_sign
    return y


# --- 1, 2: standing and the mount --------------------------------------------------------------------------------------


@dataclasses.dataclass
class Standing:
    gravity: np.ndarray  # mean accel, right-handed IMU frame, in g
    ankle_deg: float


def standing(log: DlLog, side: str, window: tuple[float, float]) -> Standing:
    rows = (log.t >= window[0]) & (log.t <= window[1])
    if rows.sum() < 50:
        raise ValueError(f"only {rows.sum()} rows in the standing window {window}")
    accel, gyro = site_vectors(log.sent[rows], side)
    if np.abs(gyro).max() > 10.0 or accel.std(axis=0).max() > 0.02:
        raise ValueError(
            f"the standing window {window} is not quiet: |gyro| up to {np.abs(gyro).max():.0f} deg/s, accel sd up to "
            f"{accel.std(axis=0).max():.3f} g"
        )
    return Standing(gravity=accel.mean(axis=0), ankle_deg=float(log.sent[rows, 6].mean()))


def swing_axis(log: DlLog, side: str, walk: np.ndarray, *, min_rate: float = 100.0):
    """Principal axis of the gyro during swing (logged is_stance 0, |gyro| > ``min_rate`` deg/s), right-handed IMU frame,
    pointing the way forward swing turns (+z on the planar model); and each axis' share of the second moment."""
    _, gyro = site_vectors(log.sent, side)
    rows = walk & (log.is_stance == 0) & (np.linalg.norm(gyro, axis=1) > min_rate)
    if rows.sum() < 100:
        raise ValueError(f"only {rows.sum()} swing rows to find the gyro axis from")
    moments, axes = np.linalg.eigh(gyro[rows].T @ gyro[rows] / rows.sum())
    axis = axes[:, -1] * np.sign(axes[2, -1])
    return axis, moments[::-1] / moments.sum()


def kabsch(pairs, weights=None) -> np.ndarray:
    """The rotation R minimizing sum w |v - R u|^2 over (u, v) pairs."""
    weights = np.ones(len(pairs)) if weights is None else weights
    b = sum(w * np.outer(v, u) for (u, v), w in zip(pairs, weights))
    u_, _, vt = np.linalg.svd(b)
    return u_ @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u_ @ vt))]) @ vt


def fit_mount(gravity: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
    """The mount that makes the keyframe read ``gravity``'s direction and planar swing turn about ``axis``.

    Returns the site -> IMU rotation (BootSensorFrontEnd's mount matrix) and the ``mount_rpy_deg`` that gives it.
    """
    to_imu = kabsch([((0.0, 1.0, 0.0), gravity / np.linalg.norm(gravity)), ((0.0, 0.0, 1.0), axis / np.linalg.norm(axis))])
    return to_imu, rpy_deg_from_rotation(to_imu.T)


def mean_rotation(rotations) -> np.ndarray:
    """The rotation closest to the average of ``rotations`` (chordal L2 mean)."""
    return kabsch([(e, r @ e) for r in rotations for e in np.eye(3)])


# --- 3: stride profiles ------------------------------------------------------------------------------------------------


def stride_profiles(t, x, strikes, *, points: int = 101, shortest: float = 0.8, longest: float = 1.6) -> np.ndarray:
    """[stride, point, channel]: each stride between consecutive strikes, resampled to ``points`` from 0 to 100%."""
    idx = np.flatnonzero(strikes)
    out = []
    for a, b in zip(idx[:-1], idx[1:]):
        if shortest <= t[b] - t[a] <= longest:
            u = np.linspace(t[a], t[b], points)
            out.append(np.stack([np.interp(u, t[a : b + 1], x[a : b + 1, c]) for c in range(x.shape[1])], axis=1))
    if not out:
        raise ValueError("no strides between the strikes")
    return np.array(out)


def profile_metrics(sim: np.ndarray, log: np.ndarray) -> dict[str, dict[str, float]]:
    """Per channel, sim's mean stride against the log's ([point, channel] each)."""
    out = {}
    last = len(sim) - 1
    for c, name in enumerate(INPUT_CHANNELS):
        s, g = sim[:, c], log[:, c]
        out[name] = dict(
            r=float(np.corrcoef(s, g)[0, 1]) if s.std() > 1e-9 and g.std() > 1e-9 else float("nan"),
            p2p_ratio=float(np.ptp(s) / np.ptp(g)) if np.ptp(g) > 0 else float("nan"),
            mean_diff=float(s.mean() - g.mean()),
            peak_shift=float((np.argmax(s) - np.argmax(g)) / last),
            trough_shift=float((np.argmin(s) - np.argmin(g)) / last),
        )
    return out


def replay_reference(
    model, front_ends: dict, reference=REFERENCE, *, start: float = 10.0, seconds: float = 60.0, decimals: int | None = 5
):
    """The reference gait set kinematically on ``model`` at the boot's rate; each side's 8 channels through its front end,
    rounded to ``decimals`` as the Pi sends them (None: not rounded).

    Every model joint the reference has (by name) follows it; qacc is the derivative of the reference's dq.
    """
    import mujoco

    ref = np.load(reference, allow_pickle=True)
    series, fs = ref["series_data"].item(), float(ref["metadata"].item()["sample_rate"])
    i0, i1 = int(start * fs), int((start + seconds) * fs)
    if i1 > len(series["q_pelvis_tx"]):
        raise ValueError(f"the reference has {len(series['q_pelvis_tx']) / fs:.1f} s; asked for {start + seconds:.1f}")
    t_ref = np.arange(i1 - i0) / fs
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    key_q = data.qpos.copy()
    joints = []
    for j in range(model.njnt):
        name = model.joint(j).name
        if f"q_{name}" in series and model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            dq = np.asarray(series[f"dq_{name}"][i0:i1], float)
            joints.append(
                (
                    int(model.jnt_qposadr[j]),
                    int(model.jnt_dofadr[j]),
                    np.asarray(series[f"q_{name}"][i0:i1], float),
                    dq,
                    np.gradient(dq, 1 / fs),
                )
            )
    sim = types.SimpleNamespace(data=data)
    ticks = np.arange(0.0, seconds - 1.0 / fs, 1.0 / RATE)
    channels = {side: np.empty((len(ticks), len(INPUT_CHANNELS))) for side in front_ends}
    for i, t in enumerate(ticks):
        data.qpos[:] = key_q
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        for qadr, dadr, q, dq, ddq in joints:
            data.qpos[qadr] = np.interp(t, t_ref, q)
            data.qvel[dadr] = np.interp(t, t_ref, dq)
            data.qacc[dadr] = np.interp(t, t_ref, ddq)
        data.time = t
        mujoco.mj_inverse(model, data)
        for side, front_end in front_ends.items():
            sample = front_end.sample(sim)
            channels[side][i] = sample if decimals is None else [round(v, decimals) for v in sample]  # '%.5f' on the Pi
    return ticks, channels


# --- 4: what each difference does to the network -----------------------------------------------------------------------


def synthesize(x: np.ndarray, side: str, to_imu: np.ndarray, synthesizer: OutOfPlaneFilter) -> np.ndarray:
    """The boot's 8 channels ``x`` with the out-of-plane components replaced by ``synthesizer``'s, offline.

    Into the site frame (signs and the mount ``to_imu`` undone), predict and replace site-frame accel z and gyro x and y,
    and back: what ``BootSensorFrontEnd`` does with an ``out_of_plane`` filter, before its quantization.
    """
    accel, gyro = site_vectors(x, side)
    accel_site, gyro_site = accel @ to_imu, gyro @ to_imu  # rows: to_imu.T @ v, IMU -> site
    predicted = synthesizer.predict(accel_site, gyro_site, x)
    accel_site[:, 2], gyro_site[:, 0], gyro_site[:, 1] = predicted[:, 0], predicted[:, 1], predicted[:, 2]
    return with_vectors(x, side, accel_site @ to_imu.T, gyro_site @ to_imu.T)


def perturbations(x, side, to_imu, ankle_mean, synthesizer: OutOfPlaneFilter | None = None) -> dict[str, np.ndarray]:
    """The logged inputs, and the logged inputs with one sim-vs-boot difference each."""
    out = {"unchanged": x}
    zeroed = x.copy()
    zeroed[:, list(OUT_OF_PLANE)] = 0.0
    out["out-of-plane zeroed"] = zeroed
    accel, gyro = site_vectors(x, side)
    accel_site, gyro_site = accel @ to_imu, gyro @ to_imu  # rows: to_imu.T @ v, IMU -> site
    planar_accel, planar_gyro = accel_site.copy(), gyro_site.copy()
    planar_accel[:, 2] = 0.0
    planar_gyro[:, 0:2] = 0.0
    out["mount crosstalk only"] = with_vectors(x, side, planar_accel @ to_imu.T, planar_gyro @ to_imu.T)
    if synthesizer is not None:
        out["synthesized out-of-plane"] = synthesize(x, side, to_imu, synthesizer)
    for delta in (-10.0, 5.0):
        shifted = x.copy()
        shifted[:, 6] += delta
        out[f"ankle {delta:+.0f} deg"] = shifted
    scaled = x.copy()
    scaled[:, 6] = ankle_mean + 1.3 * (x[:, 6] - ankle_mean)
    scaled[:, 7] *= 1.3
    out["ankle range x1.3"] = scaled
    return out


def _edges(on):
    return np.flatnonzero(on[1:] & ~on[:-1]) + 1, np.flatnonzero(~on[1:] & on[:-1]) + 1


def _shifts(t, reference, test, tolerance: float = 0.25) -> np.ndarray:
    if not len(test):
        return np.array([])
    nearest = test[np.abs(t[test][None, :] - t[reference][:, None]).argmin(axis=1)]
    shift = t[nearest] - t[reference]
    return shift[np.abs(shift) <= tolerance]


def network_effects(t, variants: dict, weights, rows: np.ndarray) -> dict[str, dict[str, float]]:
    """Each variant through the network (one stream each), against the first, over ``rows``."""
    names = list(variants)
    net = StreamingGaitNet(weights, n_streams=len(names), heads=HEADS)
    stacked = np.stack([variants[name] for name in names], axis=1)
    outputs = {head: np.empty((len(t), len(names))) for head in HEADS}
    for i in range(len(t)):
        for head, value in net.step(stacked[i]).items():
            outputs[head][i] = value
    phase = np.clip(np.round(outputs["stance_phase"], 5), 0.0, 1.0)
    on = np.round(outputs["stance_swing"]) > 0.5
    speed = np.empty_like(outputs["velocity"])
    for j in range(len(names)):
        f = Butterworth(**SPEED_FILTER)
        speed[:, j] = [f.filter(round(float(v), 5)) for v in outputs["velocity"][:, j]]
    rises0, falls0 = (e[rows[e]] for e in _edges(on[:, 0]))
    out = {}
    for j, name in enumerate(names):
        rises, falls = (e[rows[e]] for e in _edges(on[:, j]))
        hs, to = _shifts(t, rises0, rises), _shifts(t, falls0, falls)
        out[name] = dict(
            phase_rmse=float(np.sqrt(np.mean((phase[rows, j] - phase[rows, 0]) ** 2))),
            is_stance_agreement=float(np.mean(on[rows, j] == on[rows, 0])),
            heel_strike_shift_ms=float(np.median(hs) * 1e3) if len(hs) else float("nan"),
            heel_strike_sd_ms=float(np.std(hs) * 1e3) if len(hs) else float("nan"),
            toe_off_shift_ms=float(np.median(to) * 1e3) if len(to) else float("nan"),
            extra_heel_strikes=int(len(rises) - len(hs)),
            speed_bias=float(np.mean(speed[rows, j] - speed[rows, 0])),
        )
    return out


# --- the report --------------------------------------------------------------------------------------------------------


def walking_rows(log: DlLog, config: pd.DataFrame, start: float) -> np.ndarray:
    """From ``start`` to when the boot's assistance ended (SWING_ONLY back on), or 10 s before the log's end."""
    swing_only = config.SWING_ONLY.astype(str) == "True"
    turned_on = config.loop_time[swing_only & ~swing_only.shift(fill_value=True)]
    end = float(turned_on.iloc[-1]) if len(turned_on) and turned_on.iloc[-1] > start else log.t[-1] - 10.0
    return (log.t >= start) & (log.t <= end)


def build_model(config_path=DL_CONFIG):
    import mujoco

    from myoassist_utils.compose import compose_env_model

    env_params = json.loads(pathlib.Path(config_path).read_text())["env_params"]
    return mujoco.MjModel.from_xml_string(add_boot_imus(compose_env_model(env_params["msk_key"], env_params["device_key"])))


def _print_effects(effects):
    print(
        f"   {'variant':26s} {'phase RMSE':>10s} {'is_stance':>9s} {'heel strike ms':>15s} {'toe-off ms':>10s} "
        f"{'extra HS':>8s} {'speed bias':>10s}"
    )
    for name, e in effects.items():
        print(
            f"   {name:26s} {e['phase_rmse']:10.4f} {e['is_stance_agreement']:9.2%} "
            f"{e['heel_strike_shift_ms']:+7.1f} +/- {e['heel_strike_sd_ms']:4.1f} {e['toe_off_shift_ms']:+10.1f} "
            f"{e['extra_heel_strikes']:8d} {e['speed_bias']:+10.3f}"
        )


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefixes", nargs="+", help="each session's log path up to LEFT.csv / RIGHT.csv / CONFIG.csv")
    parser.add_argument("--weights", help=".npz from tools/recover_gait_net.py; without it step 4 is skipped")
    parser.add_argument("--standing", nargs=2, type=float, default=(0.5, 4.5), metavar=("START", "END"))
    parser.add_argument("--walk-start", type=float, default=35.0, help="walking rows start here (s)")
    parser.add_argument("--config", default=str(DL_CONFIG), help="the env config whose model to replay")
    parser.add_argument("--seconds", type=float, default=60.0, help="seconds of the reference gait to replay")
    parser.add_argument("--synthesize", action="store_true", help="add a fitted out-of-plane filter to step 4")
    parser.add_argument("--save-filter", type=pathlib.Path, help="fit the out-of-plane filter on all sessions, save it here")
    parser.add_argument("--ridge", type=float, default=1e-3, help="the filter's ridge, relative to the design's diagonal")
    args = parser.parse_args(argv)
    if args.save_filter and not args.weights:
        parser.error("--save-filter needs --weights, to report the network's RMSE on each session left out")

    sessions = {}
    for prefix in args.prefixes:
        config = pd.read_csv(prefix + "CONFIG.csv")
        for side_name, side in SIDES.items():
            log = DlLog.read(prefix + f"{side_name}.csv")
            walk = walking_rows(log, config, args.walk_start)
            stand = standing(log, side, tuple(args.standing))
            axis, shares = swing_axis(log, side, walk)
            to_imu, rpy = fit_mount(stand.gravity, axis)
            sessions[prefix, side] = types.SimpleNamespace(
                side=side,
                log=log,
                walk=walk,
                stand=stand,
                axis=axis,
                shares=shares,
                to_imu=to_imu,
                rpy=rpy,
                logged_standing=float(config[f"{side_name}_STANDING_ANGLE"].iloc[-1]),
            )

    calibration = {}
    for side in SIDES.values():
        mine = [s for (_, sd), s in sessions.items() if sd == side]
        to_imu = mean_rotation([s.to_imu for s in mine])
        calibration[side] = dict(
            standing_angle_deg=float(np.mean([s.stand.ankle_deg for s in mine])),
            mount_rpy_deg=tuple(round(v, 2) for v in rpy_deg_from_rotation(to_imu.T)),
            to_imu=to_imu,
        )

    model = build_model(args.config)
    front_ends = {
        side: BootSensorFrontEnd(model, side, standing_angle_deg=c["standing_angle_deg"], mount_rpy_deg=c["mount_rpy_deg"])
        for side, c in calibration.items()
    }
    ticks, sim = replay_reference(model, front_ends, seconds=args.seconds)
    settled = ticks > 3.0
    sim_mean = {}
    for side in SIDES.values():
        strikes = replay_gyro_strikes(ticks[settled], sim[side][settled, 5], GYRO)
        sim_mean[side] = stride_profiles(ticks[settled], sim[side][settled], strikes).mean(axis=0)

    weights = load_weights(args.weights)[0] if args.weights else None
    for (prefix, side), s in sessions.items():
        name = pathlib.Path(prefix).name.rstrip("_")
        side_name = next(k for k, v in SIDES.items() if v == side)
        log = s.log
        g = s.stand.gravity
        roll, pitch, yaw = s.rpy
        print(f"\n=== {name} {side_name}: {s.walk.sum()} walking rows")
        print(
            f"1. standing   gravity {np.round(g, 3).tolist()} g ({np.degrees(np.arctan2(-g[0], g[1])):+.1f} deg forward, "
            f"{np.degrees(np.arcsin(g[2] / np.linalg.norm(g))):+.1f} deg toward +z); ankle {s.stand.ankle_deg:+.2f} deg "
            f"(CONFIG's {side_name}_STANDING_ANGLE {s.logged_standing:+.2f}, read with the cable taut)"
        )
        print(
            f"2. mount      swing gyro axis {np.round(s.axis, 3).tolist()} (second moment {np.round(s.shares, 3).tolist()}); "
            f"roll {roll:+.1f}, pitch {pitch:+.1f}, yaw {yaw:+.1f} deg"
        )
        strikes = replay_gyro_strikes(log.t[s.walk], log.sent[s.walk, 5], GYRO)
        log_mean = stride_profiles(log.t[s.walk], log.sent[s.walk], strikes).mean(axis=0)
        print("3. profiles   sim vs log mean stride:  channel  r  p2p sim/log  mean sim-log  peak, trough shift (cycle)")
        for channel, m in profile_metrics(sim_mean[side], log_mean).items():
            print(
                f"             {channel:15s} {m['r']:5.2f} {m['p2p_ratio']:6.2f} {m['mean_diff']:+9.2f} "
                f"{m['peak_shift']:+6.2f} {m['trough_shift']:+6.2f}"
            )
        if weights is None:
            continue
        synthesizer = None
        if args.synthesize:
            others = [o for (p, sd), o in sessions.items() if sd == side and p != prefix]
            if not others:
                raise ValueError("--synthesize fits the filter on the other sessions; give at least two")
            synthesizer = OutOfPlaneFilter.fit([(*_site(o), o.log.sent, o.walk) for o in others], ridge=args.ridge)
        ankle_mean = float(np.mean(log.sent[s.walk, 6]))
        effects = network_effects(log.t, perturbations(log.sent, side, s.to_imu, ankle_mean, synthesizer), weights, s.walk)
        print("4. network    each sim-vs-boot difference alone, against the unchanged logged inputs:")
        _print_effects(effects)

    if args.save_filter:
        filters = fit_filters(sessions, weights, ridge=args.ridge)
        save_out_of_plane_filters(args.save_filter, filters)
        features = ", ".join(OutOfPlaneFilter.FEATURES)
        print(f"\nOut-of-plane filter ({features}; ridge {args.ridge:g}) saved to {args.save_filter}")
        for side, f in filters.items():
            rmse = ", ".join(f"{v:.4f}" for v in f.loo_rmse)
            print(f"  {side}: stance-phase RMSE with each session left out of the fit: {rmse}")

    print("\nExoControllerParams, averaged over the sessions:")
    for side, c in calibration.items():
        roll, pitch, yaw = c["mount_rpy_deg"]
        print(f'  "ankle_standing_angle_{side}_deg": {c["standing_angle_deg"]:.2f},')
        print(
            f'  "imu_mount_{side}_roll_deg": {roll}, "imu_mount_{side}_pitch_deg": {pitch}, "imu_mount_{side}_yaw_deg": {yaw},'
        )
    return calibration


def _site(session):
    """A session's accel and gyro in its own fitted site frame."""
    accel, gyro = site_vectors(session.log.sent, session.side)
    return accel @ session.to_imu, gyro @ session.to_imu


def fit_filters(sessions: dict, weights, *, ridge: float = 1e-3) -> dict[str, OutOfPlaneFilter]:
    """Per leg, the out-of-plane filter fit on every session's walking rows (each in its own fitted site frame).

    Each carries its leave-one-out RMSE: per session, the network's stance-phase RMSE over its walking rows when a filter
    fit on the other sessions synthesizes its out-of-plane channels, against its unchanged inputs.
    """
    filters = {}
    for side in SIDES.values():
        mine = [(prefix, s) for (prefix, sd), s in sessions.items() if sd == side]
        if len(mine) < 2:
            raise ValueError("--save-filter reports each session held out of the fit; give at least two")
        loo = []
        for prefix, held_out in mine:
            fitted = OutOfPlaneFilter.fit([(*_site(o), o.log.sent, o.walk) for p, o in mine if p != prefix], ridge=ridge)
            variants = {
                "unchanged": held_out.log.sent,
                "synthesized": synthesize(held_out.log.sent, side, held_out.to_imu, fitted),
            }
            loo.append(network_effects(held_out.log.t, variants, weights, held_out.walk)["synthesized"]["phase_rmse"])
        full = OutOfPlaneFilter.fit([(*_site(s), s.log.sent, s.walk) for _, s in mine], ridge=ridge)
        filters[side] = OutOfPlaneFilter(
            full.coefficients, full.ankle_mean, full.features, lags=full.lags, rate_hz=full.rate_hz, ridge=ridge, loo_rmse=loo
        )
    return filters


if __name__ == "__main__":
    main()
