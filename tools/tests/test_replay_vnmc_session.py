"""tools/replay_vnmc_session.py, on a synthetic VNMC session log written the way the boot writes one.

Real session logs are human-subject data and stay out of the repo, so this runs the port itself, at the boot's 175 Hz, on
a made-up walking ankle angle, and logs it with the boot's columns and quirks: commanded_torque left stale outside
stance, and loop iterations that ran but were not logged (the boot writes a row only on a new sensor packet). Then the
tool must recover what was put in, exactly where the boot's own code would: a tool that mis-read a column, missed an
unlogged iteration or compared stale commands would call a faithful port broken, or a broken one faithful.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import pytest

from myoassist_utils.exo_ctrl import BootStateMachine, StrideAverageGaitPhaseEstimator
from myoassist_utils.exo_ctrl.boot_state import STANCE
from myoassist_utils.exo_ctrl.vnmc import MusculoTendonJoint, VNMCLeg, VNMCStance

RATE, STRIDE, REEL_IN = 175.0, 1.15, 0.15
DURATION, START, END = 50.0, 10.0, 48.0
QUIET_UNTIL = 5.0  # standing still before this, walking after
STANDING_DEG = -6.0
CONFIG = dict(
    STANCE_CONTROL_STYLE="StanceCtrlStyle.VIRTUALNEUROMUSCULARCONTROLLER",
    SWING_ONLY=False,
    PEAK_TORQUE=25,
    VNMC_GAIN=1.468,
    MUSCLE_UPDATE_FREQUENCY=1,
    V_MAX=6,
    L_OPT=0.04,
    L_SLACK=0.26,
    PHI_REF=0,
    E_REF=0.1,
    TARGET_FREQ=175,
    NUM_STRIDES_REQUIRED=2,
)


def _ankle(t):
    """A walking ankle angle (deg, plantarflexion-positive): dorsiflexing through stance, push-off, swing."""
    if t < QUIET_UNTIL:
        return STANDING_DEG
    phase = ((t - QUIET_UNTIL) % STRIDE) / STRIDE
    return float(np.interp(phase, [0.0, 0.08, 0.45, 0.55, 0.66, 0.8, 1.0], [-2.0, -6.0, -16.0, -10.0, 12.0, -6.0, -2.0]))


class _Strikes:
    def reset(self):
        self.strike_time = -math.inf

    def detect(self, t, signal):
        if signal:
            self.strike_time = t
        return bool(signal)


def _leg():
    return VNMCLeg(
        stance=VNMCStance(muscle=MusculoTendonJoint(timestep=1 / RATE), gain=1.468, peak_torque=25.0),
        heel_strike_detector=_Strikes(),
        phase_estimator=StrideAverageGaitPhaseEstimator(),
        state_machine=BootStateMachine(reel_in_time=REEL_IN, reel_out_time=0.2),
    )


def _boot_log(hidden_at=()):
    """One leg's log, and the rows the boot ran an unlogged iteration before. At the first row from each time in
    ``hidden_at`` it ran one more iteration first, a period after the last row and on its angle (no new packet), so that
    row comes two periods after the last."""
    leg = _leg()
    strikes = np.arange(QUIET_UNTIL + 0.3, DURATION, STRIDE)
    pending = sorted(hidden_at)
    rows, hidden, t, last_cmd, last_angle = [], set(), 0.0057, np.nan, STANDING_DEG
    for i in range(int(DURATION * RATE)):
        if pending and t >= pending[0]:
            pending.pop(0)
            hidden.add(i)
            leg.step(t, (False, last_angle))
            t += 1 / RATE
        hs = bool(np.any((strikes > t - 1 / RATE) & (strikes <= t)))
        angle = _ankle(t)
        cmd = leg.step(t, (hs, angle))
        d = leg.diagnostics()
        state = int(d["control_state"])
        if state == STANCE:
            last_cmd = cmd
        moving = t >= QUIET_UNTIL
        rows.append(
            dict(
                loop_time=t,
                state_time=1000 * t,
                ankle_angle=angle,
                mtu_force=d["mtu_force"],
                length_CE=d["length_ce"],
                velocity_CE=d["velocity_ce"],
                vnmc_torque=d["vnmc_torque"],
                m_stim=d["m_stim"],
                scalefactor=d["scalefactor"],
                commanded_torque=last_cmd,  # stale outside stance, as on the boot
                ankle_torque_from_current=cmd if state == STANCE else 0.5,
                controller=state,
                control_state=state,
                did_heel_strike=int(hs),
                gait_phase=np.nan if d["phase"] < 0 else d["phase"],
                gyro_x=0.0,
                gyro_y=0.0,
                gyro_z=200.0 * math.sin(t) if moving else 0.5,
                accel_x=0.0,
                accel_y=1.0 + (0.3 * math.sin(7 * t) if moving else 0.0),
                accel_z=0.0,
            )
        )
        last_angle = angle
        t += 1 / RATE
    return pd.DataFrame(rows), hidden


def _steady_times(log, in_stance: bool, n, margin=15):
    """``n`` times inside the window, each ``margin`` rows from the nearest start or end of stance. An unlogged
    iteration there runs in the state of the row before it, as the replay assumes; each one shifts the rows after it by
    a period, which stays under the margin."""
    stance = log.control_state.to_numpy() == STANCE
    t = log.loop_time.to_numpy()
    ok = [
        t[i]
        for i in range(margin, len(t) - margin)
        if START + 1 < t[i] < END - 1 and np.all(stance[i - margin : i + margin] == in_stance)
    ]
    return ok[:: max(1, len(ok) // n)][:n]


def _write(d, log, config):
    prefix = str(d / "20990101_0000_TEST_")
    for side in ("LEFT", "RIGHT"):
        log.to_csv(prefix + f"{side}.csv", index=False)
    pd.DataFrame([{"loop_time": 0.0, **config}]).to_csv(prefix + "CONFIG.csv", index=False)
    return prefix


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    plain, _ = _boot_log()
    log, hidden = _boot_log(_steady_times(plain, True, 6) + _steady_times(plain, False, 6))
    assert len(hidden) == 12
    return _write(tmp_path_factory.mktemp("vnmc"), log, CONFIG), hidden


def _load(prefix):
    from tools.replay_vnmc_session import VnmcLog, session_params

    return VnmcLog.read(prefix + "LEFT.csv"), session_params(pd.read_csv(prefix + "CONFIG.csv"), START, END)


def test_the_muscle_and_the_unlogged_iterations_are_recovered(session):
    from tools.replay_vnmc_session import MUSCLE_COLUMNS, replay_muscle

    prefix, hidden = session
    log, params = _load(prefix)
    out, found = replay_muscle(log, params)
    assert set(np.flatnonzero(found)) == hidden and found.max() == 1, "each unlogged iteration, where it was"
    for j, column in enumerate(MUSCLE_COLUMNS):
        np.testing.assert_array_equal(out[:, j], log.muscle[column], err_msg=column)
    plain, _ = replay_muscle(log, params, find_hidden=False)
    assert not np.array_equal(plain[:, 3], log.muscle["vnmc_torque"]), "the unlogged iterations matter"


def test_reflex_scaling_and_toe_off_are_exact(session):
    from tools.replay_vnmc_session import _runs, replay_muscle, replay_stance, toe_off_on_logged_torque

    log, params = _load(session[0])
    _, hidden = replay_muscle(log, params)
    out = replay_stance(log, params, hidden)
    in_stance = log.state == STANCE
    np.testing.assert_array_equal(out["stim"], log.stim)
    np.testing.assert_array_equal(out["scalefactor"][in_stance], log.scalefactor[in_stance])
    np.testing.assert_array_equal(out["command"][in_stance], log.commanded[in_stance])
    last = [b for a, b in _runs(log.state, STANCE)]
    assert len(last) >= 30 and np.all(out["toe_off"][last] == 1) and out["toe_off"].sum() == len(last)
    # On the logged torque alone the rule misses the unlogged iterations' torques, so it holds only in the stances
    # without one -- which is why the replay steps the muscle through them.
    offsets = toe_off_on_logged_torque(log, START, END)
    runs = [(a, b) for a, b in _runs(log.state, STANCE) if START <= log.t[a] <= END]
    clean = np.array([not np.any(hidden[a : b + 2]) for a, b in runs])
    assert clean.sum() >= 25 and np.all(offsets[clean] == 0)


def test_the_leg_reproduces_the_states_with_the_boots_reel_in_and_nearly_with_a_fixed_one(session):
    from tools.replay_vnmc_session import REEL_IN as REEL_IN_STATE
    from tools.replay_vnmc_session import durations, logged_reel_ins, replay_leg, replay_muscle

    log, params = _load(session[0])
    _, hidden = replay_muscle(log, params)
    window = (log.t >= START) & (log.t <= END)
    exact = replay_leg(log, params, hidden, reel_in_time=0.3, reel_ins=logged_reel_ins(log))
    np.testing.assert_array_equal(exact["state"][window], log.state[window])
    np.testing.assert_array_equal(exact["command"][window], log.assist_command()[window])
    reel = durations(log.t, log.state, REEL_IN_STATE, START, END)
    assert reel.mean() == pytest.approx(REEL_IN, abs=1 / RATE)
    fixed = replay_leg(log, params, hidden, reel_in_time=reel.mean())
    assert np.mean(fixed["state"][window] == log.state[window]) > 0.98
    impulse = fixed["command"][window].sum() / log.assist_command()[window].sum()
    assert impulse == pytest.approx(1.0, abs=0.02)


def test_stale_commands_outside_stance_do_not_count(session):
    log, _ = _load(session[0])
    assert np.isfinite(log.commanded[log.state != STANCE]).any(), "the synthetic log does leave them stale"
    assert np.all(log.assist_command()[log.state != STANCE] == 0.0)


@pytest.mark.parametrize("rate", [150.0, RATE])
def test_in_loop_tracks_the_boot_rate_replay(session, rate):
    """On ticks of the env's schedule, at 150 Hz (3-step delay, a 1/150 s Euler step) or the boot's 175: the torque
    lags by under a tick, its size within a few percent."""
    from tools.replay_exoboot_log import stride_stats
    from tools.replay_vnmc_session import replay_in_loop, replay_leg, replay_muscle

    log, params = _load(session[0])
    _, hidden = replay_muscle(log, params)
    reference = replay_leg(log, params, hidden, reel_in_time=REEL_IN)
    loop = replay_in_loop(log, params, reel_in_time=REEL_IN, rate=rate)
    lag, peak, impulse = stride_stats(log, reference["command"], loop["t"], loop["command"], START, END)
    assert len(lag) >= 30
    assert abs(np.nanmedian(lag)) < 1e3 / rate, f"lag {np.nanmedian(lag):.2f} ms"
    assert abs(np.nanmedian(peak)) < 0.03 and abs(np.nanmedian(impulse)) < 0.05


def test_the_standing_angle_needs_a_quiet_window(session):
    from tools.replay_vnmc_session import standing_angle

    log, _ = _load(session[0])
    assert standing_angle(log, (0.5, 4.5)) == pytest.approx(STANDING_DEG)
    with pytest.raises(ValueError, match="not quiet"):
        standing_angle(log, (6.0, 9.0))


def test_the_session_must_be_the_vnmc_with_constant_parameters_over_the_window(session):
    from tools.replay_vnmc_session import session_params

    config = pd.read_csv(session[0] + "CONFIG.csv")
    params = session_params(config, START, END)
    assert (params.gain, params.rate, params.num_strides_required) == (1.468, 175.0, 2)
    with pytest.raises(ValueError, match="did not run the VNMC"):
        session_params(config.assign(STANCE_CONTROL_STYLE="StanceCtrlStyle.FOURPOINTSPLINE"), START, END)
    ramp = pd.concat([config, config.assign(loop_time=20.0, PEAK_TORQUE=20)], ignore_index=True)
    with pytest.raises(ValueError, match="change inside the window"):
        session_params(ramp, START, END)
    assert session_params(ramp, 21.0, END).at(np.array([5.0, 25.0]), ramp.PEAK_TORQUE.to_numpy()).tolist() == [25, 20]
    with pytest.raises(ValueError, match="SWING_ONLY"):
        session_params(config.assign(SWING_ONLY=True), START, END)


def test_runs_end_to_end(session, capsys):
    from tools.replay_vnmc_session import main

    main([session[0], "--start", str(START), "--end", str(END)])
    out = capsys.readouterr().out
    for layer in ("0. standing", "1. muscle", "2. reflex", "3. scaling", "4. leg", "5. impulse", "6. in-loop 150 Hz"):
        assert layer in out, layer
    assert "12 unlogged iterations found at 12 of 12 gaps" in out
    suggested = float(re.search(r"reel_in_time ([0-9.]+) \(both legs\)", out).group(1))
    assert suggested == pytest.approx(REEL_IN, abs=1 / RATE)
    assert f"ankle_standing_angle_l_deg {STANDING_DEG:.2f}" in out
