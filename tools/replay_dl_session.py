"""Replay an ExoBoot DL-task session log (the 4-headed network, as in mle_config) through the port, layer by layer.

In the DL task (``Task.WALKJOGDLGAITPHASE``) the Pi sends each leg's 8 sensor channels to a Jetson every loop and uses
the network's newest reply: gait phase = 0.6 x the stance-phase head, heel strike and toe-off = the rising and falling
edges of the stance/swing head (thresholded at 0.5), and the four-point spline as the stance controller. Each layer here
is fed the boot's own logged inputs, so a mismatch points at one layer:

  1. gait events  logged did_heel_strike / did_toe_off vs the edges of the logged is_stance, and vs the ported gyro
                  detector on gyro_z, which the committed code reads as if it overrode them
  2. network      the recovered network on the logged channels as the Pi sent them ('%.5f'), from a zero-filled
                  200-sample buffer, one row stale -> the logged dl_stance_phase / 0.6, dl_is_stance, dl_velocity,
                  dl_ramp. Rows it misses are retried against older buffers, which is what a stale reply looks like
  3. spline       logged gait_phase -> commanded_torque on the spline's own rows
  4. states       the ported state machine, on the logged events, gait phase and SWING_ONLY -> the logged control_state.
                  The boot ends reel-in and reel-out on cable slack or a timeout; the port uses the log's mean durations
  5. torque       the port's torque (the spline in stance, nothing elsewhere) -> the boot's stance command

The log is the boot's per-side CSV: 40 named columns, then six unnamed ones once the Jetson replies (``TRAILING``).

    python tools/replay_dl_session.py "<session dir>/<date>_<time>_<subject>_<trial>_" --weights gait_net_4headed.npz \\
        --start 35 --end 490

Pick the window as for tools/replay_exoboot_log.py: constant parameters, starting 5 s or more after the last change.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses

import numpy as np
import pandas as pd

from myoassist_utils.exo_ctrl import BootStateMachine, FourPointSpline
from myoassist_utils.exo_ctrl.boot_state import REEL_IN, REEL_OUT, STANCE
from myoassist_utils.exo_ctrl.gait_net import HEADS, INPUT_CHANNELS, StreamingGaitNet, load_weights

try:
    from tools.replay_exoboot_log import FOUR_POINT_SPLINE_CONTROLLER, params_in_window, replay_gyro_strikes, strike_offsets
except ModuleNotFoundError:  # run as a script, with tools/ itself on sys.path
    from replay_exoboot_log import FOUR_POINT_SPLINE_CONTROLLER, params_in_window, replay_gyro_strikes, strike_offsets

TRAILING = ("dl_stance_phase", "dl_is_stance", "dl_velocity", "dl_ramp", "velocity", "ramp")
GAIT_PHASE_SCALE = 0.6  # DLGAITSTATEESTIMATOR: gait_phase = 0.6 x stance phase
IS_STANCE_THRESHOLD = 0.5
MATCH = 1e-3  # a reply "matches" when every compared head is within this (the log keeps 5 decimals; FP16 adds ~1e-4)
MAX_LAG = 3


@dataclasses.dataclass
class DlLog:
    t: np.ndarray
    sent: np.ndarray  # [n, 8] the channels as the Pi sent them, rounded to 5 decimals
    gyro: np.ndarray
    heel_strike: np.ndarray
    toe_off: np.ndarray
    is_stance: np.ndarray  # the Jetson's is_stance as received (0/1), NaN before the first reply
    stance_phase: np.ndarray  # dl_stance_phase / 0.6: the stance-phase head as received, clipped to [0, 1]
    velocity: np.ndarray
    ramp: np.ndarray
    gait_phase: np.ndarray
    commanded: np.ndarray
    controller: np.ndarray
    state: np.ndarray  # control_state

    @classmethod
    def read(cls, path: str) -> DlLog:
        with open(path, newline="") as f:
            header = next(csv.reader(f))
        names = header + [name for name in TRAILING if name not in header]
        df = pd.read_csv(path, header=None, skiprows=1, names=names, float_precision="round_trip")
        missing = sorted(set(INPUT_CHANNELS + TRAILING[:4]) - set(df.columns))
        if missing:
            raise ValueError(f"{path} lacks {missing}; is it a DL-task log?")
        return cls(
            t=df.loop_time.to_numpy(),
            sent=np.round(df[list(INPUT_CHANNELS)].to_numpy(dtype=float), 5),
            gyro=df.gyro_z.to_numpy(dtype=float),
            heel_strike=df.did_heel_strike.to_numpy() == 1,
            toe_off=df.did_toe_off.to_numpy() == 1,
            is_stance=df.dl_is_stance.to_numpy(dtype=float),
            stance_phase=df.dl_stance_phase.to_numpy(dtype=float) / GAIT_PHASE_SCALE,
            velocity=df.dl_velocity.to_numpy(dtype=float),
            ramp=df.dl_ramp.to_numpy(dtype=float),
            gait_phase=df.gait_phase.to_numpy(dtype=float),
            commanded=df.commanded_torque.to_numpy(dtype=float),
            controller=df.controller.to_numpy(),
            state=df.control_state.to_numpy(),
        )


def stance_edges(is_stance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rising and falling edges of is_stance > 0.5, as DLGAITSTATEESTIMATOR turns them into heel strikes and toe-offs."""
    on = np.nan_to_num(is_stance, nan=0.0) > IS_STANCE_THRESHOLD
    rises, falls = np.zeros_like(on), np.zeros_like(on)
    rises[1:] = on[1:] & ~on[:-1]
    falls[1:] = ~on[1:] & on[:-1]
    if on[0]:
        rises[0] = True
    return rises, falls


def network_outputs(log: DlLog, weights) -> dict[str, np.ndarray]:
    """Each head after each row was pushed into a zero-filled buffer: the reply the Jetson computes from that buffer."""
    net = StreamingGaitNet(weights, n_streams=1, heads=HEADS)
    out = {head: np.empty(len(log.t)) for head in HEADS}
    for i, sample in enumerate(log.sent):
        for head, value in net.step(sample[None]).items():
            out[head][i] = value[0]
    return out


def _reply_error(log: DlLog, i: int, reply: dict[str, np.ndarray]) -> float:
    """How far a candidate reply is from the one logged on row i: the worst head, is_stance counting as 1."""
    if np.isnan(log.stance_phase[i]):
        return 0.0
    return max(
        abs(min(1.0, max(0.0, round(float(reply["stance_phase"][0]), 5))) - log.stance_phase[i]),
        abs(round(float(reply["velocity"][0]), 5) - log.velocity[i]),
        abs(round(float(reply["ramp"][0]), 5) - log.ramp[i]),
        float(round(float(reply["stance_swing"][0])) != log.is_stance[i]),
    )


def replay_sent_stream(
    log: DlLog,
    weights,
    *,
    loop_hz: float = 175.0,
    fit_rows: int = 8,
    max_resent: int = 6,
):
    """The reply the Jetson computed before each logged row, from the samples the Pi actually sent.

    The Pi sends a sample on every loop iteration but writes a row only when the actpack has new data
    (``ONLY_LOG_IF_NEW``), so a gap of several loop periods between rows can hide iterations that sent the previous
    sample again. At each gap the number of resent copies, 0 up to the gap in loop periods, is the one whose replies
    over the next ``fit_rows`` rows match the logged ones best. Returns the expected reply per row (each head) and the
    copies fitted per row.
    """
    net = StreamingGaitNet(weights, n_streams=1, heads=HEADS)
    n = len(log.t)
    expected = {head: np.full(n, np.nan) for head in HEADS}
    resent = np.zeros(n, dtype=int)
    periods = np.r_[1.0, np.diff(log.t) * loop_hz]
    reply = net.step(np.zeros((1, len(INPUT_CHANNELS))))  # placeholder; overwritten before it is ever recorded
    net.reset()
    for i in range(n):
        if i > 0 and periods[i] > 1.25:
            # Rows after this gap until the next one: the only replies that can tell the candidates apart.
            stop = i + 1
            while stop < min(n, i + fit_rows) and periods[stop] <= 1.25:
                stop += 1
            saved, best = net.snapshot(), (np.inf, 0)
            for copies in range(0, min(max_resent, int(round(periods[i])) - 1) + 1):
                net.restore(saved)
                trial = reply
                for _ in range(copies):
                    trial = net.step(log.sent[i - 1][None])
                score = 0.0
                for j in range(i, stop):
                    score += _reply_error(log, j, trial)
                    trial = net.step(log.sent[j][None])
                if score < best[0]:
                    best = (score, copies)
            net.restore(saved)
            resent[i] = best[1]
            for _ in range(best[1]):
                reply = net.step(log.sent[i - 1][None])
        for head in HEADS:
            expected[head][i] = reply[head][0]
        reply = net.step(log.sent[i][None])
    return expected, resent


def reply_errors(log: DlLog, expected: dict[str, np.ndarray], lag: int = 0) -> dict[str, np.ndarray]:
    """Per row, how far ``expected`` (one reply per row), taken ``lag`` rows earlier, is from the logged reply."""

    def shifted(x):
        if not lag:
            return x
        y = np.full(len(x), np.nan)
        y[lag:] = x[:-lag]
        return y

    stance_phase = np.clip(np.round(shifted(expected["stance_phase"]), 5), 0.0, 1.0)
    return dict(
        stance_phase=np.abs(stance_phase - log.stance_phase),
        velocity=np.abs(np.round(shifted(expected["velocity"]), 5) - log.velocity),
        ramp=np.abs(np.round(shifted(expected["ramp"]), 5) - log.ramp),
        is_stance=np.round(shifted(expected["stance_swing"])) != log.is_stance,
    )


def matches(errors: dict[str, np.ndarray]) -> np.ndarray:
    """The reply matches on the heads the boot uses (stance phase, is_stance) and on velocity. Ramp is reported on its
    own: it is logged only, and its residual is larger (its output is the largest, about -1 to 5)."""
    return (errors["stance_phase"] <= MATCH) & (errors["velocity"] <= MATCH) & ~errors["is_stance"]


def state_durations(log: DlLog, state: int, window: np.ndarray) -> np.ndarray:
    """How long each run of ``state`` lasted on the boot: its first row to the next state's first row."""
    on = (log.state == state) & window
    starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
    ends = np.flatnonzero(on & ~np.r_[on[1:], False]) + 1
    keep = ends < len(log.t)
    return log.t[ends[keep]] - log.t[starts[keep]]


def replay_state_machine(log: DlLog, swing_only: np.ndarray, spline: FourPointSpline, *, reel_in_time, reel_out_time):
    """The ported state machine, row by row, on the boot's own gait events, phase and swing_only."""
    machine = BootStateMachine(reel_in_time=reel_in_time, reel_out_time=reel_out_time)
    states, torques = np.zeros(len(log.t), dtype=int), np.zeros(len(log.t))
    for i, t in enumerate(log.t):
        phase = None if np.isnan(log.gait_phase[i]) else float(log.gait_phase[i])
        states[i] = machine.step(
            float(t),
            did_heel_strike=bool(log.heel_strike[i]),
            did_toe_off=bool(log.toe_off[i]),
            gait_phase=phase,
            swing_only=bool(swing_only[i]),
        )
        torques[i] = spline.torque(phase) if states[i] == STANCE else 0.0
    return states, torques


def swing_only_by_row(config: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    """SWING_ONLY as the boot had it on each row: the last CONFIG row at or before it."""
    rows = np.searchsorted(config.loop_time.to_numpy(), t, side="right") - 1
    values = config.SWING_ONLY.astype(str).to_numpy() == "True"
    return np.where(rows >= 0, values[rows.clip(min=0)], True)


def report_side(name: str, log: DlLog, params, weights, start: float, end: float, swing_only=None) -> None:
    window = (log.t >= start) & (log.t <= end)
    print(f"\n=== {name}: {window.sum()} rows in [{start:g}, {end:g}] s ===")

    rises, falls = stance_edges(log.is_stance)
    for label, logged, edges in (
        ("heel strikes", log.heel_strike, rises),
        ("toe-offs", log.toe_off, falls),
    ):
        offsets, extra = strike_offsets(edges, logged, window)
        on_row = f"{np.mean(offsets == 0):.1%}" if len(offsets) else "n/a"
        print(
            f"1. events     logged {label} ({len(offsets)}) on an is_stance edge's own row: {on_row}; "
            f"{extra} edges with no logged event"
        )
    offsets, extra = strike_offsets(replay_gyro_strikes(log.t, log.gyro, params.gyro), log.heel_strike, window)
    on_row = f"{np.mean(offsets == 0):.1%}" if len(offsets) else "n/a"
    print(f"             logged heel strikes on a gyro-detector strike's row: {on_row} ({extra} extra)")

    if weights is not None:
        replied = window & ~np.isnan(log.stance_phase)
        naive = matches(reply_errors(log, network_outputs(log, weights), 1))
        expected, resent = replay_sent_stream(log, weights)
        e = reply_errors(log, expected)
        ok = matches(e)
        print(
            f"2. network    over {replied.sum()} replies: stance phase |err| p50 {np.median(e['stance_phase'][replied]):.1e} "
            f"p99 {np.percentile(e['stance_phase'][replied], 99):.1e}; velocity p99 "
            f"{np.percentile(e['velocity'][replied], 99):.1e}; ramp p99 {np.percentile(e['ramp'][replied], 99):.1e}; "
            f"is_stance agrees {1 - np.mean(e['is_stance'][replied]):.3%}"
        )
        unexplained = replied & ~ok
        stale = {}
        for lag in range(1, MAX_LAG):
            hit = unexplained & matches(reply_errors(log, expected, lag))
            stale[lag] = int(hit.sum())
            unexplained &= ~hit
        gaps = resent[window] > 0
        print(
            f"             stance phase, velocity and is_stance all within {MATCH:g} on {np.mean(ok[replied]):.3%} of "
            f"replies (from the logged rows alone, without the resent samples: {np.mean(naive[replied]):.3%}); "
            f"{gaps.sum()} gaps hid {resent[window].sum()} resent samples; of the misses, "
            + ", ".join(f"{k} are the reply from {lag} row(s) before (stale)" for lag, k in stale.items())
            + f", {unexplained.sum()} unexplained; ramp within {MATCH:g} on {np.mean(e['ramp'][replied] <= MATCH):.1%}"
        )

    spline = FourPointSpline(**params.spline)
    rows = window & (log.controller == FOUR_POINT_SPLINE_CONTROLLER)
    err = np.array([spline.torque(p) for p in log.gait_phase[rows]]) - log.commanded[rows]
    print(f"3. spline     max |port - commanded_torque| = {np.abs(err).max():.1e} N*m over {rows.sum()} spline rows")

    if swing_only is None:
        return
    reel_in, reel_out = state_durations(log, REEL_IN, window), state_durations(log, REEL_OUT, window)
    states, torques = replay_state_machine(
        log, swing_only, spline, reel_in_time=float(np.mean(reel_in)), reel_out_time=float(np.mean(reel_out))
    )
    agree = np.mean(states[window] == log.state[window])
    boot_assist = np.where(log.state == STANCE, np.nan_to_num(log.commanded), 0.0)
    dt = np.median(np.diff(log.t))
    impulse_port, impulse_boot = torques[window].sum() * dt, boot_assist[window].sum() * dt
    diff = (torques - boot_assist)[window]
    print(
        f"4. states     reel-in {np.mean(reel_in) * 1e3:.0f} +/- {np.std(reel_in) * 1e3:.0f} ms, reel-out "
        f"{np.mean(reel_out) * 1e3:.0f} +/- {np.std(reel_out) * 1e3:.0f} ms on the boot (slack or timeout; the port uses "
        f"the means); ported state machine on the boot's events: same state on {agree:.2%} of rows"
    )
    print(
        f"5. torque     impulse port {impulse_port:.1f}, boot command {impulse_boot:.1f} N*m*s "
        f"({100 * (impulse_port / impulse_boot - 1):+.1f}%); rms diff {np.sqrt(np.mean(diff**2)):.2f} N*m, rows off by > 1 N*m "
        f"{np.mean(np.abs(diff) > 1):.2%}"
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefix", help="log path up to LEFT.csv / RIGHT.csv / CONFIG.csv")
    parser.add_argument(
        "--weights",
        default=None,
        help=".npz from tools/recover_gait_net.py; without it layer 2 is skipped",
    )
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    args = parser.parse_args(argv)

    config = pd.read_csv(args.prefix + "CONFIG.csv")
    params = params_in_window(config, args.start, args.end)
    weights = load_weights(args.weights)[0] if args.weights else None
    print(f"parameters: {params.spline}; gyro detector {params.gyro}")
    for side in ("LEFT", "RIGHT"):
        log = DlLog.read(args.prefix + f"{side}.csv")
        report_side(side, log, params, weights, args.start, args.end, swing_only=swing_only_by_row(config, log.t))


if __name__ == "__main__":
    main()
