"""Replay an ExoBoot VNMC session log through the ported VNMC, one layer at a time, to check the port against the boot.

The boot logs, on every loop iteration that read a new sensor packet, the ankle angle, its muscle's states (mtu_force,
length_CE, velocity_CE: normalized), the muscle's torque (vnmc_torque), its stimulation (m_stim), the scalefactor, the
commanded torque and the control state. Each layer is fed the boot's own inputs, so a mismatch points at one layer:

  1. muscle    logged ankle angle and m_stim in        -> mtu_force, length_CE, velocity_CE and vnmc_torque
  2. reflex    the port's own muscle, logged states    -> m_stim: 0.01 + VNMC_GAIN x the force 4 iterations earlier in
                                                          stance, 0.01 outside it
  3. scaling   the same run                            -> scalefactor and commanded_torque in stance; the toe-off on the
                                                          stance's last row
  4. leg       logged heel strikes and ankle angle in  -> control_state: the whole leg, its toe-off, reel-out's 0.2 s
                                                          timer, and a fixed reel-in (measured here, per side)
  5. impulse   the same run                            -> impulse per stride against the boot's command and its measured
                                                          torque
  6. in-loop   the device env's tick schedule (150 Hz by default) inside 1200 Hz physics, the logged ankle angle read at
               each tick and each strike seen on the first tick at or after it -> lag, peak, impulse and stance end
               against the boot-rate replay of layer 4. At 150 Hz the muscle's Euler step is 1/150 s and its 20 ms
               afferent delay 3 steps (4 at 175 Hz, 22.9 ms).

Iterations the boot did not log. It writes a row only when the actpack has a new packet, but runs its muscle on every
loop iteration. At a gap in loop_time it may have run iterations it did not log, on the previous packet's angle. The
muscle shows how many: layer 1 tries 0 up to the gap's length in loop periods at each gap, and keeps the count that
reproduces the next row; every later layer replays the same iterations.

Usage (the prefix is everything before LEFT.csv / RIGHT.csv / CONFIG.csv):

    python tools/replay_vnmc_session.py "<session dir>/<date>_<time>_<subject>_<trial>_" --start 31 --end 485

The replay always runs from the log's first row (the muscle and the scaling carry history); only the comparison is
windowed. Pick a window with constant parameters: after the PEAK_TORQUE ramp, before SWING_ONLY is set at the end.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import math

import numpy as np
import pandas as pd

from myoassist_utils.exo_ctrl import BootStateMachine, StrideAverageGaitPhaseEstimator
from myoassist_utils.exo_ctrl.boot_state import REEL_IN, REEL_OUT, STANCE, SWING
from myoassist_utils.exo_ctrl.vnmc import (
    IDLE_STIMULATION,
    VNMC_REEL_OUT_TIME,
    MusculoTendonJoint,
    TorqueToeOffDetector,
    VNMCLeg,
    VNMCStance,
)
from tools.replay_exoboot_log import LoggedHeelStrikes, in_loop_ticks, stride_stats, strides

STATE_NAMES = {REEL_OUT: "ReelOut", SWING: "Swing", REEL_IN: "ReelIn", STANCE: "Stance"}
VNMC_STYLE = "StanceCtrlStyle.VIRTUALNEUROMUSCULARCONTROLLER"
# CONFIG.csv column -> MusculoTendonJoint argument
MUSCLE_PARAMS = {"L_OPT": "l_opt", "V_MAX": "v_max", "L_SLACK": "l_slack", "E_REF": "e_ref", "PHI_REF": "phi_ref_deg"}
MUSCLE_COLUMNS = ("mtu_force", "length_CE", "velocity_CE", "vnmc_torque")
# A quiet standing: what calibrate_boot_sensors.py asks of one on the DL sessions.
QUIET_GYRO, QUIET_ACCEL_SD = 10.0, 0.02


@dataclasses.dataclass(frozen=True)
class SessionParams:
    rate: float  # TARGET_FREQ: the boot's loop, and its muscle's step
    gain: float
    num_strides_required: int
    muscle: dict
    # The parameters that change during the session, as (loop_time, value) steps.
    config_times: np.ndarray
    peak_torque: np.ndarray
    swing_only: np.ndarray

    def at(self, t: np.ndarray, values: np.ndarray) -> np.ndarray:
        return values[np.clip(np.searchsorted(self.config_times, t, side="right") - 1, 0, None)]


def session_params(config: pd.DataFrame, start: float, end: float) -> SessionParams:
    """The session's parameters, refusing a session that is not the VNMC throughout or a window they change in."""
    if (config.STANCE_CONTROL_STYLE.astype(str) != VNMC_STYLE).any():
        raise ValueError("this session did not run the VNMC throughout")
    if (config.MUSCLE_UPDATE_FREQUENCY != 1).any():
        raise ValueError("the boot's muscle can only run with MUSCLE_UPDATE_FREQUENCY 1")
    watched = ["PEAK_TORQUE", "VNMC_GAIN", "SWING_ONLY", *MUSCLE_PARAMS, "TARGET_FREQ", "NUM_STRIDES_REQUIRED"]
    rows = config[watched].astype(str)
    changed = config.loop_time[(rows != rows.shift()).any(axis=1)].iloc[1:]
    inside = changed[(changed > start) & (changed < end)]
    if len(inside):
        raise ValueError(f"parameters change inside the window, at {inside.round(3).tolist()} s")
    before = config[config.loop_time <= start]
    if before.empty:
        raise ValueError("the window starts before the first config row")
    row = before.iloc[-1]
    if str(row.SWING_ONLY) == "True":
        raise ValueError("SWING_ONLY is set over the window, so the boot applied no stance torque")
    return SessionParams(
        rate=float(row.TARGET_FREQ),
        gain=float(row.VNMC_GAIN),
        num_strides_required=int(row.NUM_STRIDES_REQUIRED),
        muscle={arg: float(row[column]) for column, arg in MUSCLE_PARAMS.items()},
        config_times=config.loop_time.to_numpy(dtype=float),
        peak_torque=config.PEAK_TORQUE.to_numpy(dtype=float),
        swing_only=config.SWING_ONLY.astype(str).to_numpy() == "True",
    )


@dataclasses.dataclass
class VnmcLog:
    t: np.ndarray
    angle: np.ndarray  # ankle_angle, deg, plantarflexion-positive
    stim: np.ndarray
    muscle: dict  # MUSCLE_COLUMNS -> array
    scalefactor: np.ndarray
    commanded: np.ndarray  # stale outside stance: the boot does not clear it
    measured: np.ndarray  # ankle_torque_from_current
    state: np.ndarray
    heel_strike: np.ndarray
    imu: np.ndarray | None = None  # gyro_x/y/z (deg/s) then accel_x/y/z (g), for the standing check

    @classmethod
    def read(cls, path: str) -> VnmcLog:
        # The boot writes repr-exact floats; read them back exactly, for bit-for-bit comparisons.
        df = pd.read_csv(path, float_precision="round_trip")
        missing = [c for c in ("m_stim", "scalefactor", *MUSCLE_COLUMNS) if c not in df.columns]
        if missing:
            raise ValueError(f"{path} has no VNMC columns {missing} (DO_INCLUDE_VNMC_DATA was off)")
        imu_columns = [f"{kind}_{axis}" for kind in ("gyro", "accel") for axis in "xyz"]
        return cls(
            t=df.loop_time.to_numpy(dtype=float),
            angle=df.ankle_angle.to_numpy(dtype=float),
            stim=df.m_stim.to_numpy(dtype=float),
            muscle={c: df[c].to_numpy(dtype=float) for c in MUSCLE_COLUMNS},
            scalefactor=df.scalefactor.to_numpy(dtype=float),
            commanded=df.commanded_torque.to_numpy(dtype=float),
            measured=df.ankle_torque_from_current.to_numpy(dtype=float),
            state=df.control_state.to_numpy(),
            heel_strike=df.did_heel_strike.to_numpy() == 1,
            imu=df[imu_columns].to_numpy(dtype=float) if set(imu_columns) <= set(df.columns) else None,
        )

    @property
    def strikes(self) -> np.ndarray:
        return self.t[self.heel_strike]

    def assist_command(self) -> np.ndarray:
        """What the boot commanded as assist: the VNMC's torque in stance, nothing in the other states."""
        return np.where(self.state == STANCE, np.nan_to_num(self.commanded), 0.0)


def standing_angle(log: VnmcLog, window: tuple[float, float]) -> float:
    """The mean ankle angle over a quiet standing, refusing a window in which the leg moves."""
    rows = (log.t >= window[0]) & (log.t <= window[1])
    if rows.sum() < 50:
        raise ValueError(f"only {rows.sum()} rows in the standing window {window}")
    if log.imu is not None:
        gyro, accel = log.imu[rows, :3], log.imu[rows, 3:]
        if np.abs(gyro).max() > QUIET_GYRO or accel.std(axis=0).max() > QUIET_ACCEL_SD:
            raise ValueError(
                f"the standing window {window} is not quiet: |gyro| up to {np.abs(gyro).max():.0f} deg/s, accel sd up "
                f"to {accel.std(axis=0).max():.3f} g"
            )
    return float(log.angle[rows].mean())


def _muscle(params: SessionParams, rate: float | None = None) -> MusculoTendonJoint:
    return MusculoTendonJoint(timestep=1.0 / (rate or params.rate), **params.muscle)


def _hidden_stimulation(muscle: MusculoTendonJoint, state, gain: float) -> float:
    """What an unlogged iteration stimulated with: the reflex in stance, else the idle stimulation."""
    return IDLE_STIMULATION + gain * muscle.sensory_force if state == STANCE else IDLE_STIMULATION


def replay_muscle(log: VnmcLog, params: SessionParams, *, find_hidden: bool = True):
    """Layer 1: the muscle on the logged angle and m_stim. Returns (its four logged quantities per row, the number of
    unlogged iterations found before each row). At a row more than 1.5 loop periods after the last, 0 up to the gap's
    length in periods of unlogged iterations are tried, on the last row's angle, and the count that reproduces this
    row's mtu_force and length_CE best is kept."""
    period = 1.0 / params.rate
    muscle = _muscle(params)
    out = np.zeros((len(log.t), len(MUSCLE_COLUMNS)))
    hidden = np.zeros(len(log.t), dtype=int)
    logged_f, logged_l = log.muscle["mtu_force"], log.muscle["length_CE"]
    for i in range(len(log.t)):
        gap = log.t[i] - log.t[i - 1] if i else 0.0
        counts = range(int(math.ceil(gap / period)) + 1) if find_hidden and gap > 1.5 * period else (0,)
        best = None
        for n in counts:
            m = copy.deepcopy(muscle) if len(counts) > 1 else muscle
            for _ in range(n):
                m.step(_hidden_stimulation(m, log.state[i - 1], params.gain), math.radians(log.angle[i - 1]))
            m.step(log.stim[i], math.radians(log.angle[i]))
            force, length, _ = m.states()
            err = abs(force - logged_f[i]) + abs(length - logged_l[i])
            if best is None or err < best[0]:
                best = (err, n, m)
        _, hidden[i], muscle = best
        out[i] = (*muscle.states(), muscle.torque)
    return out, hidden


def _iterations(log: VnmcLog, hidden: np.ndarray):
    """Every loop iteration the boot ran, in order: (time, row it is logged on or -1, angle, row whose state it ran in,
    whether it carries the row's heel strike). Unlogged ones are spread evenly over their gap."""
    for i in range(len(log.t)):
        for j in range(hidden[i]):
            t = log.t[i - 1] + (j + 1) * (log.t[i] - log.t[i - 1]) / (hidden[i] + 1)
            yield t, -1, log.angle[i - 1], i - 1, False
        yield log.t[i], i, log.angle[i], i, bool(log.heel_strike[i])


def replay_stance(log: VnmcLog, params: SessionParams, hidden: np.ndarray) -> dict[str, np.ndarray]:
    """Layers 2-3: ``VNMCStance`` stepped in the logged states, its own muscle and reflex. Per row: stimulation,
    scalefactor, command, and whether its toe-off fired there or on an unlogged iteration after it (which ran in that
    row's state)."""
    stance = VNMCStance(muscle=_muscle(params), gain=params.gain, peak_torque=float(params.peak_torque[0]))
    out = {key: np.zeros(len(log.t)) for key in ("stim", "scalefactor", "command", "toe_off")}
    previous = None
    for t, row, angle, state_row, _ in _iterations(log, hidden):
        stance.peak_torque = float(params.at(np.array([t]), params.peak_torque)[0])
        state = log.state[state_row]
        if state == STANCE:
            if previous != STANCE:
                stance.start_stance()
            stance.stance_tick(angle)
        else:
            stance.idle_tick(angle)
        previous = state
        out["toe_off"][state_row] = max(out["toe_off"][state_row], float(stance.toe_off))
        if row >= 0:
            out["stim"][row], out["scalefactor"][row], out["command"][row] = (
                stance.stimulation,
                stance.scalefactor,
                stance.command,
            )
    return out


def make_leg(params: SessionParams, *, reel_in_time: float, rate: float | None = None) -> VNMCLeg:
    """The port's whole leg as the env builds it, but reporting the strikes it is given."""
    return VNMCLeg(
        stance=VNMCStance(muscle=_muscle(params, rate), gain=params.gain, peak_torque=float(params.peak_torque[0])),
        heel_strike_detector=LoggedHeelStrikes(),
        phase_estimator=StrideAverageGaitPhaseEstimator(num_strides_required=params.num_strides_required),
        state_machine=BootStateMachine(reel_in_time=reel_in_time, reel_out_time=VNMC_REEL_OUT_TIME),
    )


def _step_leg(leg: VNMCLeg, params: SessionParams, t: float, strike: bool, angle: float) -> tuple[float, float, float]:
    leg.stance.peak_torque = float(params.at(np.array([t]), params.peak_torque)[0])
    leg.swing_only = bool(params.at(np.array([t]), params.swing_only)[0])
    torque = leg.step(t, (strike, angle))
    return torque, float(leg.state_machine.state), leg.stance.muscle.torque


def logged_reel_ins(log: VnmcLog) -> np.ndarray:
    """Per row: how long the boot's reel-in lasted if one starts there (to its first stance row), else NaN."""
    out = np.full(len(log.t), np.nan)
    for a, b in _runs(log.state, REEL_IN):
        if b + 1 < len(log.t) and log.state[b + 1] == STANCE:
            out[a] = log.t[b + 1] - log.t[a]
    return out


def replay_leg(
    log: VnmcLog,
    params: SessionParams,
    hidden: np.ndarray,
    *,
    reel_in_time: float,
    reel_ins: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Layer 4: the whole leg on the logged strikes and angle, every iteration the boot ran. Per row: command, control
    state, raw muscle torque. With ``reel_ins`` (``logged_reel_ins``), each reel-in ends where the boot's did instead
    of after ``reel_in_time``: the boot ends it on cable slack, which the log does not carry."""
    leg = make_leg(params, reel_in_time=reel_in_time)
    out = {key: np.zeros(len(log.t)) for key in ("command", "state", "raw")}
    for t, row, angle, _, strike in _iterations(log, hidden):
        if reel_ins is not None and row >= 0 and np.isfinite(reel_ins[row]):
            # Its timer then runs out (strictly after its duration) on the boot's first stance row.
            leg.state_machine._reel_in.delay_time = reel_ins[row] - 1e-9
        result = _step_leg(leg, params, t, strike, angle)
        if row >= 0:
            out["command"][row], out["state"][row], out["raw"][row] = result
    return out


def replay_in_loop(log: VnmcLog, params: SessionParams, *, reel_in_time: float, rate: float) -> dict[str, np.ndarray]:
    """Layer 6: the leg on the device env's ticks, at ``rate``: each tick reads the logged angle interpolated to that
    instant (the sim's ankle is continuous; holding the last row instead would hand some ticks a repeated angle, a
    plateau in the torque that the toe-off rule reads as its rise), and sees each logged strike on the first tick at or
    after it. Per tick: time, command, state, raw muscle torque."""
    ticks = in_loop_ticks(log.t[-1], rate)
    ticks = ticks[ticks >= log.t[0]]
    seen = np.zeros(len(ticks), dtype=bool)
    seen[np.searchsorted(ticks, log.strikes, side="left").clip(max=len(ticks) - 1)] = True
    angle = np.interp(ticks, log.t, log.angle)
    leg = make_leg(params, reel_in_time=reel_in_time, rate=rate)
    rows = np.array([_step_leg(leg, params, t, s, a) for t, s, a in zip(ticks, seen, angle)])
    return {"t": ticks, "command": rows[:, 0], "state": rows[:, 1], "raw": rows[:, 2]}


# --- measurements ----------------------------------------------------------------------------------------------------


def _runs(state: np.ndarray, value) -> list[tuple[int, int]]:
    """(first, last) index of each run of ``state == value``."""
    on = np.concatenate([[False], state == value, [False]])
    edges = np.flatnonzero(np.diff(on.astype(int)))
    return list(zip(edges[::2], edges[1::2] - 1))


def durations(t: np.ndarray, state: np.ndarray, value, start: float, end: float) -> np.ndarray:
    """How long each complete run of ``value`` inside the window lasted: from its first row to the next state's."""
    out = []
    for a, b in _runs(state, value):
        if b + 1 < len(t) and start <= t[a] and t[b + 1] <= end:
            out.append(t[b + 1] - t[a])
    return np.array(out)


def toe_off_on_logged_torque(log: VnmcLog, start: float, end: float) -> np.ndarray:
    """The toe-off rule run on each logged stance's own vnmc_torque: rows from where it fires to the stance's last row
    (0 = it ends on that row; NaN = the rule never fired)."""
    out = []
    torque = log.muscle["vnmc_torque"]
    for a, b in _runs(log.state, STANCE):
        if not (start <= log.t[a] <= end):
            continue
        detector, peaks = TorqueToeOffDetector(), np.maximum.accumulate(torque[a : b + 1])
        fired = next((i for i, peak in zip(range(a, b + 1), peaks) if detector.step(torque[i], peak)), None)
        out.append(np.nan if fired is None else b - fired)
    return np.array(out, dtype=float)


def stance_ends(t: np.ndarray, state: np.ndarray, strikes: np.ndarray, start: float, end: float) -> np.ndarray:
    """Per logged stride in the window, when stance first ended (NaN if it did not)."""
    out = []
    for a, b in strides(strikes, start, end):
        m = (t >= a) & (t < b)
        s, ts = state[m], t[m]
        ended = np.flatnonzero((s[:-1] == STANCE) & (s[1:] != STANCE))
        out.append(ts[ended[0] + 1] if len(ended) else np.nan)
    return np.array(out)


def stride_peaks(t: np.ndarray, raw: np.ndarray, strikes: np.ndarray, start: float, end: float) -> np.ndarray:
    return np.array([raw[(t >= a) & (t < b)].max() for a, b in strides(strikes, start, end)])


def _diff_stats(port: np.ndarray, logged: np.ndarray) -> str:
    d = np.abs(port - logged)
    return f"exact {np.mean(d == 0):.2%}, p99 {np.percentile(d, 99):.1e}, max {d.max():.1e}"


def report_side(
    name: str,
    log: VnmcLog,
    params: SessionParams,
    start: float,
    end: float,
    *,
    reel_in_time: float | None,
    in_loop_rate: float,
    standing: tuple[float, float],
) -> dict:
    window = (log.t >= start) & (log.t <= end)
    n_strides = len(strides(log.strikes, start, end))
    print(f"\n=== {name}: {n_strides} strides in [{start:g}, {end:g}] s ===")
    angle = standing_angle(log, standing)
    print(f"0. standing   quiet standing {standing[0]:g}-{standing[1]:g} s: ankle {angle:+.2f} deg")

    out, hidden = replay_muscle(log, params)
    gaps = window & (np.diff(log.t, prepend=log.t[0]) > 1.5 / params.rate)
    plain, _ = replay_muscle(log, params, find_hidden=False)
    print(
        f"1. muscle     {int(hidden[window].sum())} unlogged iterations found at {int(np.count_nonzero(hidden[gaps]))} of "
        f"{int(gaps.sum())} gaps; with them:"
    )
    for j, column in enumerate(MUSCLE_COLUMNS):
        print(f"               {column:12s} {_diff_stats(out[window, j], log.muscle[column][window])}")
    d = np.abs(plain[window, 3] - log.muscle["vnmc_torque"][window])
    print(f"               without them, vnmc_torque p99 {np.percentile(d, 99):.1e}, max {d.max():.1e} N*m")

    stance = replay_stance(log, params, hidden)
    in_stance = window & (log.state == STANCE)
    print(f"2. reflex     m_stim on all rows: {_diff_stats(stance['stim'][window], log.stim[window])}")
    print(f"3. scaling    scalefactor in stance: {_diff_stats(stance['scalefactor'][in_stance], log.scalefactor[in_stance])}")
    print(f"              commanded_torque in stance: {_diff_stats(stance['command'][in_stance], log.commanded[in_stance])}")
    last = np.array([b for a, b in _runs(log.state, STANCE) if start <= log.t[a] <= end])
    fired = stance["toe_off"] == 1
    print(
        f"              the port's toe-off fires on the stance's last row in {np.mean(fired[last]):.2%} of {len(last)} "
        f"stances, and on {int(np.count_nonzero(fired[in_stance])) - int(np.count_nonzero(fired[last]))} other rows"
    )
    offsets = toe_off_on_logged_torque(log, start, end)
    print(
        f"              on the logged vnmc_torque alone, the rule fires on the last stance row in "
        f"{np.mean(offsets == 0):.2%} (offsets {pd.Series(offsets).value_counts(dropna=False).to_dict()})"
    )

    reel_in = durations(log.t, log.state, REEL_IN, start, end)
    reel_out = durations(log.t, log.state, REEL_OUT, start, end)
    used = reel_in.mean() if reel_in_time is None else reel_in_time
    print(
        f"4. leg        reel-in (boot) {reel_in.mean() * 1e3:.0f} +/- {reel_in.std() * 1e3:.0f} ms "
        f"[{reel_in.min() * 1e3:.0f}, {reel_in.max() * 1e3:.0f}], reel-out (boot) {reel_out.mean() * 1e3:.1f} +/- "
        f"{reel_out.std() * 1e3:.1f} ms; with reel-out's {VNMC_REEL_OUT_TIME:g} s timer and"
    )
    ends_boot = stance_ends(log.t, log.state, log.strikes, start, end)
    for label, reel_ins in (("the boot's own reel-in ends", logged_reel_ins(log)), (f"reel_in_time {used * 1e3:.0f} ms", None)):
        leg = replay_leg(log, params, hidden, reel_in_time=used, reel_ins=reel_ins)
        agree = np.mean(leg["state"][window] == log.state[window])
        d_end = (stance_ends(log.t, leg["state"], log.strikes, start, end) - ends_boot) * 1e3
        print(
            f"              {label + ':':29s} control_state agrees on {agree:.2%} of rows; stance ends on the boot's row in "
            f"{np.nanmean(d_end == 0):.1%} of strides ({np.nanmean(d_end):+.1f} +/- {np.nanstd(d_end):.1f} ms)"
        )
    # The rest compares the leg as the env runs it, with a fixed reel-in.
    ends_port = stance_ends(log.t, leg["state"], log.strikes, start, end)
    for s in (REEL_IN, STANCE, REEL_OUT, SWING):
        m = window & (log.state == s)
        if m.any():
            print(
                f"              measured torque in {STATE_NAMES[s]:>7}: mean {log.measured[m].mean():5.2f}, max "
                f"{log.measured[m].max():5.2f} N*m"
            )

    boot = log.assist_command()
    dt = np.median(np.diff(log.t))
    imp_port, imp_boot, imp_meas = (x[window].sum() * dt / n_strides for x in (leg["command"], boot, log.measured))
    diff = (leg["command"] - boot)[window]
    print(
        f"5. impulse    per stride: port {imp_port:.3f}, boot command {imp_boot:.3f} ({100 * (imp_port / imp_boot - 1):+.1f}%), "
        f"boot measured {imp_meas:.3f} N*m*s; rms diff {np.sqrt(np.mean(diff**2)):.2f} N*m, rows off by > 1 N*m "
        f"{np.mean(np.abs(diff) > 1):.2%}"
    )

    results = {"reel_in": reel_in, "standing": angle}
    reference_peaks = stride_peaks(log.t, leg["raw"], log.strikes, start, end)
    for rate in dict.fromkeys((in_loop_rate, params.rate)):
        loop = replay_in_loop(log, params, reel_in_time=used, rate=rate)
        with np.errstate(invalid="ignore", divide="ignore"):  # a stride the in-loop leg left unassisted has no lag
            lag, peak, impulse = stride_stats(log, leg["command"], loop["t"], loop["command"], start, end)
        d_end = (stance_ends(loop["t"], loop["state"], log.strikes, start, end) - ends_port) * 1e3
        raw = stride_peaks(loop["t"], loop["raw"], log.strikes, start, end) / reference_peaks - 1
        print(
            f"6. in-loop {rate:g} Hz (delay {_muscle(params, rate).delay_steps} ticks): lag {np.nanmean(lag):+.1f} +/- "
            f"{np.nanstd(lag):.1f} ms, peak {np.nanmean(peak):+.1%} +/- {np.nanstd(peak):.1%}, impulse "
            f"{np.nanmean(impulse):+.1%} +/- {np.nanstd(impulse):.1%}; stance ends {np.nanmean(d_end):+.1f} +/- "
            f"{np.nanstd(d_end):.1f} ms (within a tick in {np.nanmean(np.abs(d_end) <= 1e3 / rate):.0%}); raw muscle peak "
            f"{np.nanmean(raw):+.1%} +/- {np.nanstd(raw):.1%} vs the boot-rate replay"
        )
        # The means are pulled by the odd stride whose toe-off fires on another rise of the torque, or not at all, so
        # that its stance runs through swing: count those apart.
        print(
            f"              median impulse {np.nanmedian(impulse):+.1%}; {int(np.sum(np.abs(impulse) > 0.1))} of "
            f"{len(impulse)} strides off by more than 10%"
        )
        results[rate] = dict(lag=lag, peak=peak, impulse=impulse, stance_end=d_end, raw_peak=raw)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefix", help="log path up to LEFT.csv / RIGHT.csv / CONFIG.csv")
    parser.add_argument("--start", type=float, required=True, help="window start, loop_time seconds")
    parser.add_argument("--end", type=float, required=True, help="window end, loop_time seconds")
    parser.add_argument("--reel-in-time", type=float, default=None, help="seconds; default: measured per side")
    parser.add_argument("--in-loop-rate", type=float, default=150.0, help="the device controller's rate, Hz")
    parser.add_argument("--standing", nargs=2, type=float, default=(0.5, 4.5), metavar=("START", "END"))
    args = parser.parse_args(argv)

    params = session_params(pd.read_csv(args.prefix + "CONFIG.csv"), args.start, args.end)
    print(
        f"parameters: VNMC_GAIN {params.gain:g}, PEAK_TORQUE {params.at(np.array([args.start]), params.peak_torque)[0]:g}, "
        f"TARGET_FREQ {params.rate:g}, NUM_STRIDES_REQUIRED {params.num_strides_required}, muscle {params.muscle}"
    )
    sides = {
        side: report_side(
            side,
            VnmcLog.read(args.prefix + f"{side}.csv"),
            params,
            args.start,
            args.end,
            reel_in_time=args.reel_in_time,
            in_loop_rate=args.in_loop_rate,
            standing=tuple(args.standing),
        )
        for side in ("LEFT", "RIGHT")
    }
    reel = np.concatenate([s["reel_in"] for s in sides.values()])
    print(
        f"\nexo_controller_params: reel_in_time {reel.mean():.3f} (both legs), reel_out_time {VNMC_REEL_OUT_TIME:g}, "
        f"ankle_standing_angle_r_deg {sides['RIGHT']['standing']:.2f}, ankle_standing_angle_l_deg "
        f"{sides['LEFT']['standing']:.2f}"
    )


if __name__ == "__main__":
    main()
