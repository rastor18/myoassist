"""Replay an ExoBoot session log through the ported four-point spline controller, to check the port against the boot.

The boot logs, every loop, the heel strikes its gyro detector found, the gait phase it estimated, the torque the
spline commanded, and which state it was in. That lets each layer of the port be checked on its own -- feeding it
the boot's own inputs, so a mismatch points at one layer:

  1. spline      logged phase in            -> must equal commanded_torque on rows the spline commanded
  2. estimator   logged heel strikes in     -> must equal the logged gait_phase
  3. gates       logged phase and states    -> stance must end at TOE_OFF_FRACTION; reel-in is measured
  4. controller  logged heel strikes in     -> gated torque vs the boot's command (spline in stance, zero otherwise)
  5. gyro        logged gyro_z in           -> the ported gyro detector's strikes vs the logged ones
  6. in-loop     the device env's tick schedule (150 Hz by default) inside 1200 Hz physics, with the logged strikes and
                 then with the gyro port reading gyro_z at each tick -> timing and size vs the boot-rate replay

Layers 5 and 6 need the log's gyro_z column for the gyro port; its parameters come from CONFIG.csv, or the boot's
defaults where it does not record them.

Usage (the prefix is everything before LEFT.csv / RIGHT.csv / CONFIG.csv):

    python tools/replay_exoboot_log.py "<session dir>/<date>_<time>_<subject>_<trial>_" --start 35 --end 240

Pick a window at constant speed and constant parameters, and start it at least 5 s after the last parameter
change: the boot cross-fades old and new splines over 5 s, which the port deliberately does not reproduce.
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import pandas as pd

from myoassist_utils.exo_ctrl import (
    ExoBootFourPointSplineController,
    FourPointSpline,
    GyroHeelStrikeDetector,
    StrideAverageGaitPhaseEstimator,
    TickSchedule,
)

# NeuMoveExoBoot/constants.py
REEL_OUT, SWING, REEL_IN, STANCE = 1, 2, 3, 4
STATE_NAMES = {REEL_OUT: "ReelOut", SWING: "Swing", REEL_IN: "ReelIn", STANCE: "Stance"}
FOUR_POINT_SPLINE_CONTROLLER = 8  # ControllerUsed.FourPointSplineController
FADE_DURATION = 5.0  # GenericSplineController's default cross-fade on a parameter change, in seconds
PHYSICS_RATE = 1200.0  # the RL env's physics_sim_framerate
# CONFIG.csv column -> GyroHeelStrikeDetector argument, with the boot's defaults (config_util.ConfigurableConstants).
GYRO_PARAMS = {
    "HS_GYRO_THRESHOLD": ("threshold", 100.0),
    "HS_GYRO_FILTER_N": ("filter_order", 2),
    "HS_GYRO_FILTER_WN": ("filter_cutoff_hz", 3.0),
    "TARGET_FREQ": ("sample_rate_hz", 175.0),
    "HS_GYRO_DELAY": ("delay", 0.05),
}


@dataclasses.dataclass(frozen=True)
class SessionParams:
    spline: dict
    toe_off_fraction: float
    num_strides_required: int
    gyro: dict = dataclasses.field(default_factory=lambda: {arg: default for arg, default in GYRO_PARAMS.values()})
    gyro_defaulted: tuple = ()  # CONFIG columns that were missing, so the boot's default stands in


def params_in_window(config: pd.DataFrame, start: float, end: float) -> SessionParams:
    """The parameters in effect over [start, end], refusing a window they change in."""
    if (config.STANCE_CONTROL_STYLE.astype(str) != "StanceCtrlStyle.FOURPOINTSPLINE").any():
        raise ValueError("this session did not run the four-point spline throughout")
    changes = config.loop_time[(config.loop_time > start) & (config.loop_time < end)]
    if len(changes):
        raise ValueError(f"parameters change inside the window, at {changes.round(3).tolist()} s")
    before = config[config.loop_time <= start]
    if before.empty:
        raise ValueError("the window starts before the first config row")
    if start - before.loop_time.iloc[-1] < FADE_DURATION:
        print(
            f"warning: the last parameter change ({before.loop_time.iloc[-1]:.3f} s) is within {FADE_DURATION:g} s of the "
            "window start, so the boot may still be cross-fading splines"
        )
    row = before.iloc[-1]
    if str(row.SWING_ONLY) == "True":
        raise ValueError("SWING_ONLY is set over the window, so the boot applied no stance torque")
    spline = dict(
        rise_fraction=float(row.RISE_FRACTION),
        peak_fraction=float(row.PEAK_FRACTION),
        fall_fraction=float(row.FALL_FRACTION),
        peak_torque=float(row.PEAK_TORQUE),
        bias_torque=float(row.SPLINE_BIAS),
    )
    gyro, defaulted = {}, []
    for column, (arg, default) in GYRO_PARAMS.items():
        if column in row.index and pd.notna(row[column]):
            gyro[arg] = type(default)(row[column])
        else:
            gyro[arg] = default
            defaulted.append(column)
    return SessionParams(
        spline,
        float(row.TOE_OFF_FRACTION),
        int(row.NUM_STRIDES_REQUIRED),
        gyro,
        tuple(defaulted),
    )


class LoggedHeelStrikes:
    """A heel-strike detector that reports the strikes the boot logged, so the rest of the port sees its inputs."""

    def reset(self) -> None:
        self.strike_time = float("-inf")

    def detect(self, t: float, signal: float) -> bool:
        if signal:
            self.strike_time = t
        return bool(signal)


@dataclasses.dataclass
class Log:
    t: np.ndarray
    heel_strike: np.ndarray
    phase: np.ndarray  # NaN where the boot had no phase
    state: np.ndarray
    controller: np.ndarray
    commanded: np.ndarray  # stale outside spline rows: the boot does not clear it
    measured: np.ndarray  # ankle_torque_from_current
    gyro: np.ndarray | None = None  # gyro_z, deg/s, if the log has it

    @classmethod
    def read(cls, path: str) -> Log:
        # The boot writes repr-exact floats; read them back exactly. pandas' default parser can be an ulp off, which is
        # enough to move a filtered gyro peak by a sample where two neighbours are nearly equal.
        df = pd.read_csv(path, float_precision="round_trip")
        return cls(
            t=df.loop_time.to_numpy(),
            heel_strike=df.did_heel_strike.to_numpy() == 1,
            phase=df.gait_phase.to_numpy(dtype=float),
            state=df.control_state.to_numpy(),
            controller=df.controller.to_numpy(),
            commanded=df.commanded_torque.to_numpy(dtype=float),
            measured=df.ankle_torque_from_current.to_numpy(dtype=float),
            gyro=df.gyro_z.to_numpy(dtype=float) if "gyro_z" in df.columns else None,
        )

    @property
    def strikes(self) -> np.ndarray:
        return self.t[self.heel_strike]

    def assist_command(self) -> np.ndarray:
        """What the boot commanded as assist: the spline's torque in stance, nothing in the other states."""
        return np.where(self.state == STANCE, np.nan_to_num(self.commanded), 0.0)


def strides(strikes: np.ndarray, start: float, end: float):
    s = strikes[(strikes >= start) & (strikes <= end)]
    return list(zip(s[:-1], s[1:]))


def check_spline(log: Log, params: SessionParams, window: np.ndarray) -> np.ndarray:
    rows = window & (log.controller == FOUR_POINT_SPLINE_CONTROLLER)
    spline = FourPointSpline(**params.spline)
    return np.array([spline.torque(p) for p in log.phase[rows]]) - log.commanded[rows]


def replay_phase(log: Log, params: SessionParams) -> np.ndarray:
    estimator = StrideAverageGaitPhaseEstimator(num_strides_required=params.num_strides_required)
    out = np.full(len(log.t), np.nan)
    for i, (t, hs) in enumerate(zip(log.t, log.heel_strike)):
        phase = estimator.estimate(t, hs)
        if phase is not None:
            out[i] = phase
    return out


def reel_in_durations(log: Log, start: float, end: float) -> np.ndarray:
    """Time from each heel strike to the boot entering stance."""
    out = []
    for a, b in strides(log.strikes, start, end):
        m = (log.t >= a) & (log.t < b)
        state, t = log.state[m], log.t[m]
        if state[0] == REEL_IN and (state == STANCE).any():
            out.append(t[np.argmax(state == STANCE)] - t[0])
    return np.array(out)


def toe_off_offsets(log: Log, params: SessionParams, start: float, end: float) -> np.ndarray:
    """Rows between the first phase past toe-off and the first reel-out row, per stride (0 = exact)."""
    out = []
    for a, b in strides(log.strikes, start, end):
        m = (log.t >= a) & (log.t < b)
        state, phase = log.state[m], log.phase[m]
        if (state == REEL_OUT).any() and (phase > params.toe_off_fraction).any():
            out.append(int(np.argmax(state == REEL_OUT)) - int(np.argmax(phase > params.toe_off_fraction)))
    return np.array(out)


def replay_controller(
    t,
    signal,
    params: SessionParams,
    *,
    reel_in_time: float,
    detector=None,
):
    """The port's torque at times ``t``. ``signal`` is the logged strikes, or whatever ``detector`` reads instead."""
    controller = ExoBootFourPointSplineController(
        spline=FourPointSpline(**params.spline),
        heel_strike_detector=LoggedHeelStrikes() if detector is None else detector,
        phase_estimator=StrideAverageGaitPhaseEstimator(num_strides_required=params.num_strides_required),
        toe_off_fraction=params.toe_off_fraction,
        reel_in_time=reel_in_time,
    )
    return np.array([controller.step(a, s) for a, s in zip(t, signal)])


def stride_stats(
    log: Log,
    reference: np.ndarray,
    ticks: np.ndarray,
    torque: np.ndarray,
    start: float,
    end: float,
):
    """Per logged stride, ``torque`` (held per tick) against ``reference`` (held per log row): lag of the torque
    centroid in ms, and peak and impulse ratios - 1."""
    fine = np.arange(start, end, 1 / 4000)
    ref = reference[np.searchsorted(log.t, fine, side="right") - 1]
    got = torque[np.searchsorted(ticks, fine, side="right") - 1]
    lag, peak, impulse = [], [], []
    for a, b in strides(log.strikes, start, end):
        m = (fine >= a) & (fine < b)
        r, g, tf = ref[m], got[m], fine[m]
        if r.sum() <= 0:
            continue
        lag.append((np.sum(g * tf) / g.sum() - np.sum(r * tf) / r.sum()) * 1e3)
        peak.append(g.max() / r.max() - 1)
        impulse.append(g.sum() / r.sum() - 1)
    return np.array(lag), np.array(peak), np.array(impulse)


def replay_gyro_strikes(t: np.ndarray, gyro: np.ndarray, gyro_params: dict) -> np.ndarray:
    """The ported gyro detector run on the logged gyro_z, one log row per boot loop."""
    detector = GyroHeelStrikeDetector(**gyro_params)
    return np.array([detector.detect(a, g) for a, g in zip(t, gyro)])


def strike_offsets(port: np.ndarray, logged: np.ndarray, window: np.ndarray, *, match_rows: int = 3):
    """Row offset of the nearest port strike for each logged strike in the window (0 = the same row), and the number of
    port strikes in the window with no logged strike within ``match_rows`` rows."""
    port_rows, logged_rows = (
        np.flatnonzero(port & window),
        np.flatnonzero(logged & window),
    )

    def nearest(rows, to):
        if not len(rows):
            return np.full(len(to), np.iinfo(np.int64).max // 2)
        i = np.searchsorted(rows, to).clip(1, len(rows) - 1) if len(rows) > 1 else np.zeros(len(to), dtype=int)
        before, after = rows[np.maximum(i - 1, 0)], rows[i]
        return np.where(np.abs(before - to) <= np.abs(after - to), before, after) - to

    extra = int(np.sum(np.abs(nearest(logged_rows, port_rows)) > match_rows))
    return nearest(port_rows, logged_rows), extra


def in_loop_ticks(t_end: float, rate: float, physics_rate: float = PHYSICS_RATE) -> np.ndarray:
    """Times of the device env's controller ticks over [0, t_end]: the TickSchedule's substeps of the physics."""
    schedule = TickSchedule(rate_hz=rate, physics_rate_hz=physics_rate)
    return np.flatnonzero([schedule.tick() for _ in range(int(np.ceil(t_end * physics_rate)) + 1)]) / physics_rate


def compare_in_loop(
    log: Log,
    params: SessionParams,
    start: float,
    end: float,
    *,
    reel_in_time: float,
    rate: float,
):
    """The device env's pipeline: the controller on the tick schedule inside the physics, torque held per tick.
    First with the logged strikes, each seen on the first tick at or after it; then, if the log has gyro_z,
    with the gyro port reading the latest logged sample at each tick, as a sensor read at that instant would, its
    filter designed for the tick rate. Against the same controller replayed at the boot's own rows. Returns, per
    variant, ``stride_stats``."""
    reference = replay_controller(log.t, log.heel_strike, params, reel_in_time=reel_in_time)
    ticks = in_loop_ticks(log.t[-1], rate)
    ticks = ticks[ticks >= log.t[0]]
    seen = np.zeros(len(ticks), dtype=bool)
    seen[np.searchsorted(ticks, log.strikes, side="left").clip(max=len(ticks) - 1)] = True
    results = {
        "logged strikes": stride_stats(
            log,
            reference,
            ticks,
            replay_controller(ticks, seen, params, reel_in_time=reel_in_time),
            start,
            end,
        )
    }
    if log.gyro is not None:
        gyro_at_ticks = log.gyro[np.searchsorted(log.t, ticks, side="right") - 1]
        torque = replay_controller(
            ticks,
            gyro_at_ticks,
            params,
            reel_in_time=reel_in_time,
            detector=GyroHeelStrikeDetector(**{**params.gyro, "sample_rate_hz": rate}),
        )
        results["gyro port"] = stride_stats(log, reference, ticks, torque, start, end)
    return results


def report_side(
    name: str,
    log: Log,
    params: SessionParams,
    start: float,
    end: float,
    reel_in_time: float | None,
    in_loop_rate: float = 150.0,
):
    window = (log.t >= start) & (log.t <= end)
    n_strides = len(strides(log.strikes, start, end))
    print(f"\n=== {name}: {n_strides} strides in [{start:g}, {end:g}] s ===")

    err = check_spline(log, params, window)
    print(f"1. spline     max |port - commanded_torque| = {np.abs(err).max():.1e} N*m over {len(err)} spline rows")

    phase = replay_phase(log, params)
    both = window & ~np.isnan(phase) & ~np.isnan(log.phase)
    agree = np.mean(np.isnan(phase[window]) == np.isnan(log.phase[window]))
    d = phase[both] - log.phase[both]
    print(
        f"2. estimator  phase valid/invalid agrees on {agree:.2%} of rows; |diff| p99 {np.percentile(np.abs(d), 99):.1e}, "
        f"max {np.abs(d).max():.1e} (rows where the loop stalls between stamping loop_time and estimating log a later phase)"
    )

    offsets = toe_off_offsets(log, params, start, end)
    reel = reel_in_durations(log, start, end)
    print(
        f"3. gates      stance ends on the first row past {params.toe_off_fraction:g} in {np.mean(offsets == 0):.1%} of strides; "
        f"reel-in {reel.mean() * 1e3:.0f} +/- {reel.std() * 1e3:.0f} ms [{reel.min() * 1e3:.0f}, {reel.max() * 1e3:.0f}]"
    )
    for s in (REEL_IN, STANCE, REEL_OUT, SWING):
        m = window & (log.state == s)
        if m.any():
            print(
                f"              measured torque in {STATE_NAMES[s]:>7}: mean {log.measured[m].mean():5.2f}, max {log.measured[m].max():5.2f} N*m"
            )

    used = reel.mean() if reel_in_time is None else reel_in_time
    port = replay_controller(log.t, log.heel_strike, params, reel_in_time=used)
    boot = log.assist_command()
    dt = np.median(np.diff(log.t))
    imp_port, imp_boot, imp_meas = (x[window].sum() * dt / n_strides for x in (port, boot, log.measured))
    # The port's own phase differs from the logged one by ~1e-3 (the loop_time stamp precedes the estimator's
    # clock), which moves torque by a few hundredths of a N*m on the spline's slopes. Rows off by more than 1 N*m
    # are the real disagreements: reel-in ending at a fixed time instead of when the cable is taut, and a phase
    # landing on the other side of toe-off.
    diff = (port - boot)[window]
    print(
        f"4. controller reel_in_time {used * 1e3:.0f} ms: impulse/stride port {imp_port:.3f}, boot command {imp_boot:.3f} "
        f"({100 * (imp_port / imp_boot - 1):+.1f}%), boot measured {imp_meas:.3f} N*m*s; "
        f"rms diff {np.sqrt(np.mean(diff**2)):.2f} N*m, rows off by > 1 N*m {np.mean(np.abs(diff) > 1):.2%}"
    )

    if log.gyro is not None:
        offsets, extra = strike_offsets(replay_gyro_strikes(log.t, log.gyro, params.gyro), log.heel_strike, window)
        print(
            f"5. gyro       port on logged gyro_z: {np.mean(offsets == 0):.1%} of {len(offsets)} logged strikes on the same row, "
            f"{np.mean(np.abs(offsets) <= 1):.1%} within 1 row (offsets {np.unique(offsets[np.abs(offsets) <= 3]).tolist()}); "
            f"{extra} extra"
        )
    for label, (lag, peak, impulse) in compare_in_loop(log, params, start, end, reel_in_time=used, rate=in_loop_rate).items():
        print(
            f"6. in-loop {in_loop_rate:g} Hz, {label:>14}: lag {lag.mean():+.1f} +/- {lag.std():.1f} ms, "
            f"peak {peak.mean():+.1%}, impulse {impulse.mean():+.1%} vs the boot-rate replay"
        )
    return reel


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefix", help="log path up to LEFT.csv / RIGHT.csv / CONFIG.csv")
    parser.add_argument("--start", type=float, required=True, help="window start, loop_time seconds")
    parser.add_argument("--end", type=float, required=True, help="window end, loop_time seconds")
    parser.add_argument(
        "--reel-in-time",
        type=float,
        default=None,
        help="seconds; default: measured from this log",
    )
    parser.add_argument(
        "--in-loop-rate",
        type=float,
        default=150.0,
        help="the device controller's rate, Hz (the env's default: exo_controller_params.controller_rate_hz)",
    )
    args = parser.parse_args(argv)

    params = params_in_window(pd.read_csv(args.prefix + "CONFIG.csv"), args.start, args.end)
    print(
        f"parameters: {params.spline}, toe_off_fraction {params.toe_off_fraction}, "
        f"num_strides_required {params.num_strides_required}"
    )
    defaulted = f" (boot defaults for {', '.join(params.gyro_defaulted)})" if params.gyro_defaulted else ""
    print(f"gyro detector: {params.gyro}{defaulted}")
    reel = [
        report_side(
            side,
            Log.read(args.prefix + f"{side}.csv"),
            params,
            args.start,
            args.end,
            args.reel_in_time,
            args.in_loop_rate,
        )
        for side in ("LEFT", "RIGHT")
    ]
    both = np.concatenate(reel)
    print(f"\nreel-in over both legs: {both.mean() * 1e3:.0f} ms -> exo_controller_params.reel_in_time {both.mean():.3f}")


if __name__ == "__main__":
    main()
