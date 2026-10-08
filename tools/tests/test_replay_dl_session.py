"""tools/replay_dl_session.py, on a synthetic DL-task log written the way the boot writes one.

Real session logs are human-subject data and stay out of the repo, so this builds one from a random network: the Pi's
sent samples, the Jetson's replies one iteration late, gait events from the is_stance edges, the state machine's
control_state with the spline commanding in stance, and the ragged CSV (six unnamed trailing columns once replies
arrive). It also hides loop iterations the way the boot does -- sent to the Jetson, never logged -- so the tool has to
find them to reproduce the replies after each gap.

The port's state machine stands in for the boot's when writing control_state (tools/tests/test_dl_controller.py checks
it against the boot's transitions), so layers 4 and 5 are checked here for what the tool adds: reading the states,
measuring the reel-in and reel-out durations, and replaying with their means.
"""

from __future__ import annotations

import csv
import re

import numpy as np
import pandas as pd
import pytest

from myoassist_utils.exo_ctrl import BootStateMachine, FourPointSpline
from myoassist_utils.exo_ctrl.boot_state import STANCE
from myoassist_utils.exo_ctrl.gait_net import (
    HEADS,
    INPUT_CHANNELS,
    StreamingGaitNet,
    save_weights,
)
from tools.replay_exoboot_log import FOUR_POINT_SPLINE_CONTROLLER
from tools.tests.test_gait_net import random_weights, walking_samples
from tools.tests.test_replay_exoboot_log import SPLINE, _config

N_ROWS = 1400
HIDDEN = {
    500: 2,
    900: 1,
}  # row -> loop iterations before it that re-sent the previous sample and were not logged
REEL_IN_TIME, REEL_OUT_TIME = 0.157, 0.172
NAMED = [
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


def _write_log(path, weights):
    spline = FourPointSpline(**SPLINE)
    machine = BootStateMachine(reel_in_time=REEL_IN_TIME, reel_out_time=REEL_OUT_TIME)
    samples = np.round(walking_samples(N_ROWS, seed=7), 5)
    net = StreamingGaitNet(weights)
    reply, t, last_on, last_cmd = None, 0.0057, False, 0.0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(NAMED)
        for i, sample in enumerate(samples):
            for _ in range(HIDDEN.get(i, 0)):
                reply = net.step(samples[i - 1][None])
            t += (1 + HIDDEN.get(i, 0)) / 175.0 if i else 0.0
            trailing, hs, to, phase = [], 0, 0, None
            if reply is not None:
                sp = min(1.0, max(0.0, round(float(reply["stance_phase"][0]), 5)))
                on = round(float(reply["stance_swing"][0])) == 1
                hs, to = int(on and not last_on), int(last_on and not on)
                last_on = on
                phase = 0.6 * sp
                v, r = (
                    round(float(reply["velocity"][0]), 5),
                    round(float(reply["ramp"][0]), 5),
                )
                trailing = [0.6 * sp, float(on), v, r, v, r]
            state = machine.step(t, did_heel_strike=bool(hs), did_toe_off=bool(to), gait_phase=phase, swing_only=False)
            controller = FOUR_POINT_SPLINE_CONTROLLER if state == STANCE else 2
            if state == STANCE:
                last_cmd = spline.torque(phase)
            row = [i, t, *sample, hs, "" if phase is None else phase, to, last_cmd, controller, state, *trailing]
            writer.writerow(row)
            reply = net.step(sample[None])


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    d = tmp_path_factory.mktemp("dl_session")
    weights = random_weights(seed=5)
    # Centre the stance/swing head on its own median, so the random network actually switches between stance and swing.
    net = StreamingGaitNet(weights, heads=("stance_swing",))
    probs = np.array([net.step(s[None])["stance_swing"][0] for s in np.round(walking_samples(N_ROWS, seed=7), 5)])
    weights["stance_swing"]["out/bias"] = weights["stance_swing"]["out/bias"] - np.median(np.log(probs / (1 - probs)))
    save_weights(d / "net.npz", weights)
    prefix = str(d / "20990101_0000_TEST_")
    for side in ("LEFT", "RIGHT"):
        _write_log(prefix + f"{side}.csv", weights)
    config = _config(
        HS_GYRO_THRESHOLD=100.0,
        HS_GYRO_FILTER_N=2,
        HS_GYRO_FILTER_WN=3.0,
        HS_GYRO_DELAY=0.05,
        TARGET_FREQ=175.0,
    )
    pd.DataFrame([{"loop_time": 0.0, **config}]).to_csv(prefix + "CONFIG.csv", index=False)
    return prefix, weights, d / "net.npz"


def test_reads_the_ragged_log(session):
    from tools.replay_dl_session import DlLog

    log = DlLog.read(session[0] + "LEFT.csv")
    assert np.isnan(log.stance_phase[0]) and not np.isnan(log.stance_phase[1:]).any()
    assert log.sent.shape == (N_ROWS, len(INPUT_CHANNELS))
    assert (log.state == STANCE).any() and (log.state[log.controller == FOUR_POINT_SPLINE_CONTROLLER] == STANCE).all()


def test_events_are_the_is_stance_edges(session):
    from tools.replay_dl_session import DlLog, stance_edges
    from tools.replay_exoboot_log import strike_offsets

    log = DlLog.read(session[0] + "LEFT.csv")
    rises, falls = stance_edges(log.is_stance)
    everywhere = np.ones(len(log.t), dtype=bool)
    for logged, edges in ((log.heel_strike, rises), (log.toe_off, falls)):
        offsets, extra = strike_offsets(edges, logged, everywhere)
        assert len(offsets) > 0 and np.all(offsets == 0) and extra == 0


def test_finds_the_hidden_iterations_and_reproduces_every_reply(session):
    from tools.replay_dl_session import (
        DlLog,
        matches,
        network_outputs,
        replay_sent_stream,
        reply_errors,
    )

    log = DlLog.read(session[0] + "LEFT.csv")
    expected, resent = replay_sent_stream(log, session[1])
    assert {int(i): int(n) for i, n in enumerate(resent) if n} == HIDDEN
    replied = ~np.isnan(log.stance_phase)
    errors = reply_errors(log, expected)
    assert np.all(matches(errors)[replied])
    assert np.nanmax(errors["ramp"]) <= 1e-5
    # Without the hidden iterations the replies after each gap come out wrong: that is what the fit is for.
    naive = matches(reply_errors(log, network_outputs(log, session[1]), 1))
    assert not naive[replied].all()
    assert naive[1:500].all()


def test_runs_end_to_end(session, capsys):
    from tools.replay_dl_session import main

    prefix, _, weights_path = session
    main([prefix, "--weights", str(weights_path), "--start", "0", "--end", "100"])
    out = capsys.readouterr().out
    assert "1. events" in out and "2. network" in out and "3. spline" in out
    assert re.search(r"3 resent samples", out)
    assert "max |port - commanded_torque| = 0.0e+00" in out
    # Layer 4 recovers the reel durations to within a loop period or two (the timers end on the first row past them,
    # and a hidden gap can fall inside a reel-in) and replays with their means, so only rows at a transition can differ.
    for label, duration in (("reel-in", REEL_IN_TIME), ("reel-out", REEL_OUT_TIME)):
        measured = [float(ms) for ms in re.findall(rf"{label} (\d+) \+/-", out)]
        assert len(measured) == 2 and all(duration * 1e3 <= ms <= duration * 1e3 + 2e3 / 175 for ms in measured)
    agree = [float(pct) for pct in re.findall(r"same state on ([\d.]+)% of rows", out)]
    assert len(agree) == 2 and min(agree) >= 98.0
    impulse = [float(pct) for pct in re.findall(r"\(([+-][\d.]+)%\); rms diff", out)]
    assert len(impulse) == 2 and max(abs(pct) for pct in impulse) <= 2.0


def test_layer_2_is_skipped_without_weights(session, capsys):
    from tools.replay_dl_session import main

    main([session[0], "--start", "0", "--end", "100"])
    out = capsys.readouterr().out
    assert "2. network" not in out and "3. spline" in out
    assert set(HEADS) >= {"stance_phase"}
