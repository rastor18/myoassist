"""tools/rollout_controllers.py: each check it reports must measure what it claims.

The measurements run on synthetic records with known answers: a profile averaged over uneven strides comes back, a
known delay and scale come back as that lag, peak and impulse, a held signal has no off-tick changes, and strikes are
counted as seen, missed or extra as they were placed. A check that mis-measured would pass or fail the controller for
the wrong reason. The reset the tool relies on to make episodes reproducible is checked on the real env.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from tools.tests.conftest import REPO_ROOT

RATE = 1200.0
STRIDES = np.array([1.05, 1.2, 0.95, 1.1, 1.15, 1.0, 1.1])  # uneven, so phase must be per stride


def _strikes():
    return 0.3 + np.concatenate([[0.0], np.cumsum(STRIDES)])


def _phase(t, strikes):
    k = np.clip(np.searchsorted(strikes, t, side="right") - 1, 0, len(strikes) - 2)
    return np.clip((t - strikes[k]) / np.diff(strikes)[k], 0.0, 1.0)


def test_stride_average_recovers_the_profile_on_each_strides_own_phase():
    from tools.rollout_controllers import stride_average

    t = np.arange(0.0, 10.0, 1 / RATE)
    strikes = _strikes()
    centres, rows = stride_average(t, np.sin(np.pi * _phase(t, strikes)), strikes, bins=50)
    assert rows.shape == (len(STRIDES), 50)
    np.testing.assert_allclose(rows, np.broadcast_to(np.sin(np.pi * centres), rows.shape), atol=0.01)


def test_a_known_delay_and_scale_come_back_as_lag_peak_and_impulse():
    from tools.rollout_controllers import compare_strides

    t = np.arange(0.0, 10.0, 1 / RATE)
    strikes = _strikes()
    phase = _phase(t, strikes)
    reference = np.where((phase > 0.3) & (phase < 0.6), np.sin(np.pi * (phase - 0.3) / 0.3), 0.0)
    reference[t < strikes[1]] = 0.0  # a stride without torque, as in warm-up: not compared
    delivered = 1.1 * np.interp(t - 0.005, t, reference)  # 5 ms late, 10% larger
    lags, peaks, impulses = compare_strides(t, delivered, reference, strikes)
    assert len(lags) == len(STRIDES) and np.isnan(lags[0]), "entry k is stride k; the unassisted one is NaN"
    np.testing.assert_allclose(lags[1:], 5.0, atol=0.05)
    np.testing.assert_allclose(peaks[1:], 0.1, atol=2e-3)
    np.testing.assert_allclose(impulses[1:], 0.1, atol=2e-3)


def test_a_strike_dated_more_than_a_tick_late_is_told_apart():
    from tools.rollout_controllers import dated_within

    onsets = np.array([1.0, 2.0, 3.0])
    found = np.array([1.003, 2.067, 3.0])  # within a tick, 67 ms late (a first touch the tick did not see), on time
    assert dated_within(found, onsets, 1 / 150).tolist() == [True, False, True]
    assert dated_within(np.array([]), onsets, 1 / 150).tolist() == [False, False, False]


def test_a_torque_that_changes_between_ticks_is_counted():
    from tools.rollout_controllers import changes_off_tick

    tick = np.zeros(80)
    tick[::8] = 1
    held = np.repeat(np.arange(10.0), 8)  # changes only on ticks
    assert changes_off_tick(held, tick) == 0
    held[13] += 1.0  # set between ticks, and back at the next substep
    assert changes_off_tick(held, tick) == 2


def test_strikes_are_counted_as_seen_missed_and_extra():
    from tools.rollout_controllers import strike_delays

    expected = np.array([1.0, 2.0, 3.0, 4.0])
    found = np.array([1.004, 2.0, 3.02, 5.0])  # 4 ms late, on time, later than a tick (missed), and one extra
    delays, missed, false = strike_delays(found, expected, max_delay=1 / 150)
    np.testing.assert_allclose(delays, [0.004, 0.0], atol=1e-12)
    assert missed == 2 and false.tolist() == [3.02, 5.0]


def test_contacts_split_into_stances_and_touches():
    """A scuff in swing is a contact too short to be a stance; a foot loaded at the start has no onset; and a contact
    still on at the end is a stance once it has lasted long enough."""
    from tools.rollout_controllers import contacts, split_contacts

    t = np.arange(0.0, 3.0, 1 / RATE)
    grf = np.where(t < 0.4, 400.0, 0.0)  # loaded at the start: not a strike
    grf[(t >= 1.0) & (t < 1.6)] = 600.0  # a stance
    grf[(t >= 1.9) & (t < 1.92)] = 500.0  # a 20 ms scuff in swing
    grf[t >= 2.5] = 800.0  # a stance cut off by the end of the record
    onsets, ends = contacts(t, grf, on=100.0, off=25.0, min_unload=0.05)
    np.testing.assert_allclose(onsets, [1.0, 1.9, 2.5], atol=1 / RATE)
    np.testing.assert_allclose(ends[:2], [1.6, 1.92], atol=1 / RATE)
    assert np.isnan(ends[2])
    stance_onsets, stance_ends, touches, undetermined = split_contacts(onsets, ends, t[-1])
    np.testing.assert_allclose(stance_onsets, [1.0, 2.5], atol=1 / RATE)
    np.testing.assert_allclose(touches, [1.9], atol=1 / RATE)
    assert len(undetermined) == 0
    # The same stance, had the record ended 50 ms into it, could still have been a touch.
    cut = t < 2.55
    stance_onsets, _, touches, undetermined = split_contacts(
        *contacts(t[cut], grf[cut], on=100.0, off=25.0, min_unload=0.05), t[cut][-1]
    )
    np.testing.assert_allclose(undetermined, [2.5], atol=1 / RATE)
    assert 2.5 not in np.round(stance_onsets, 3) and len(touches) == 1


def test_stance_fraction_is_contact_end_over_stride():
    from tools.rollout_controllers import stride_table

    durations, fractions = stride_table(np.array([0.0, 1.0, 2.1, 3.0]), np.array([0.6, 1.7]))
    np.testing.assert_allclose(durations, [1.0, 1.1, 0.9])
    np.testing.assert_allclose(fractions[:2], [0.6, 0.7 / 1.1])
    assert np.isnan(fractions[2]), "a stride whose contact never ended has no stance fraction"


def test_the_expected_profile_is_the_gated_spline():
    from tools.rollout_controllers import gated_profile

    params = types.SimpleNamespace(
        rise_fraction=0.278,
        peak_fraction=0.543,
        fall_fraction=0.641,
        peak_torque=25.0,
        bias_torque=0.0,
        peak_hold_time=0.0,
        reel_in_time=0.157,
        toe_off_fraction=0.6,
    )
    phase = np.array([0.1, 0.2, 0.543, 0.6, 0.61, 0.9])
    torque = gated_profile(phase, params, stride=1.1)
    assert torque[0] == 0.0, "before reel-in (0.157 s of a 1.1 s stride is phase 0.143)"
    assert torque[1] == 0.0 and torque[2] == pytest.approx(25.0)
    assert torque[3] == pytest.approx(9.5, abs=0.1), "cut at toe-off, mid-fall"
    assert torque[4] == 0.0 and torque[5] == 0.0


@pytest.fixture(scope="module")
def model():
    import mujoco

    from myoassist_utils.compose import compose_env_model

    return mujoco.MjModel.from_xml_string(compose_env_model("myolegs22", "DephyExoBoot_L1", terrain=None))


def _record(model, legs, extra_keys=(), strikes=(0.0, 1.0, 0.0)):
    """Three substeps through a ``Recorder`` around a stand-in controller whose legs report ``legs(strike)``."""
    import mujoco

    from tools.rollout_controllers import Recorder

    strike = [0.0]
    inner = types.SimpleNamespace(
        actuator_ids=(model.actuator("Exo_R").id, model.actuator("Exo_L").id),
        legs=[types.SimpleNamespace(controller=types.SimpleNamespace(diagnostics=d)) for d in legs(strike)],
        reset=lambda: None,
        compute_torque=lambda sim: (0.0, 0.0),
    )
    recorder = Recorder(inner, model, extra_keys)
    data = mujoco.MjData(model)
    for k, s in enumerate(strikes):
        strike[0], data.time = s, 0.01 * k
        recorder.compute_torque(types.SimpleNamespace(data=data))
    return recorder, recorder.take()


def test_the_recorder_reads_any_controllers_diagnostics_by_key(model):
    """A key the controller does not report is NaN; a strike with no time of its own is dated to the row reporting it;
    extra keys are recorded after the fields every record has."""
    from tools.rollout_controllers import Recorder

    def legs(strike):
        return [lambda v=v: {"heel_strike": strike[0], "phase": 0.25, "speed": v} for v in (1.1, 1.2)]

    recorder, record = _record(model, legs, extra_keys=("speed",))
    assert recorder.fields == (*Recorder.FIELDS, "speed_r", "speed_l") and set(record) == set(recorder.fields)
    np.testing.assert_array_equal(record["phase_r"], 0.25)
    assert np.all(np.isnan(record["control_state_l"])) and np.all(np.isnan(record["stride_estimate_r"]))
    np.testing.assert_array_equal(record["strike_time_r"], [np.nan, 0.01, np.nan])
    np.testing.assert_array_equal(record["speed_r"], 1.1)
    np.testing.assert_array_equal(record["speed_l"], 1.2)


def test_the_recorder_takes_a_controllers_own_strike_time(model):
    def legs(strike):
        return [lambda: {"heel_strike": strike[0], "strike_time": 0.004 if strike[0] else -1.0}] * 2

    _, record = _record(model, legs)
    np.testing.assert_array_equal(record["strike_time_l"], [np.nan, 0.004, np.nan])


def test_with_no_controller_legs_every_diagnostic_is_nan(model):
    _, record = _record(model, lambda strike: [], extra_keys=("speed",))
    for key in ("strike_r", "phase_l", "control_state_r", "speed_l"):
        assert np.all(np.isnan(record[key])), key
    assert np.all(np.isfinite(record["grf_r"]))


def test_every_case_belongs_to_one_suite_after_the_shared_exo_off_case():
    from tools.rollout_controllers import CASES, EXO_OFF, SUITES, suite_of

    assert next(iter(CASES)) == EXO_OFF and suite_of(EXO_OFF) is None
    names = [case for suite in SUITES.values() for case in suite.cases]
    assert len(names) == len(set(names)) and EXO_OFF not in names, "a case name means one thing"
    for suite in SUITES.values():
        assert (REPO_ROOT / suite.config).is_file()
        assert all(suite_of(case) is suite and CASES[case] is suite.cases[case] for case in suite.cases)


# --- the DL suite ----------------------------------------------------------------------------------------------------


def test_contact_stance_and_its_true_phase():
    from tools.rollout_controllers import in_contact, true_stance_phase

    t = np.arange(0.0, 3.0, 0.01)
    onsets, ends = np.array([0.5, 2.0]), np.array([1.1, np.nan])  # the second still on at the end
    on = in_contact(t, onsets, ends)
    assert on[(t >= 0.5) & (t < 1.1)].all() and on[t >= 2.0].all() and not on[(t < 0.5) | ((t >= 1.11) & (t < 2.0))].any()
    phase = true_stance_phase(t, onsets, ends)
    np.testing.assert_allclose(phase[(t >= 0.5) & (t < 1.1)], (t[(t >= 0.5) & (t < 1.1)] - 0.5) / 0.6)
    assert np.isnan(phase[t >= 2.0]).all(), "a contact that has not ended has no phase yet"


def test_run_lengths_leave_out_the_runs_the_record_cuts():
    from tools.rollout_controllers import run_lengths

    t = np.arange(0.0, 1.0, 0.01)
    on = (t >= 0.2) & (t < 0.5) | (t >= 0.55) & (t < 0.9)  # stance 0.3 s, a 0.05 s swing blip, stance 0.35 s
    stance, swing = run_lengths(t, on)
    np.testing.assert_allclose(stance, [0.3, 0.35])
    np.testing.assert_allclose(swing, [0.05]), "the swing before 0.2 s and after 0.9 s is cut by the record"


def test_nearest_offsets_match_either_side_and_count_missed_and_extra():
    from tools.rollout_controllers import nearest_offsets

    expected = np.array([1.0, 2.0, 3.0, 4.0])
    found = np.array([0.99, 2.02, 3.5, 4.0, 4.05])  # 10 ms early, 20 ms late, too far, on time, and one extra
    offsets, missed, extra = nearest_offsets(found, expected, 0.15)
    np.testing.assert_allclose(offsets, [-0.01, 0.02, 0.0], atol=1e-12)
    assert missed == 1 and sorted(extra.tolist()) == [3.5, 4.05]


def test_band_rms_finds_a_sine_in_its_band():
    from tools.rollout_controllers import band_rms

    t = np.arange(0.0, 20.0, 1 / 175)
    high, bands = band_rms(30.0 * np.sin(2 * np.pi * 2.0 * t) + 10.0 * np.sin(2 * np.pi * 8.0 * t) + 5.0, 175.0)
    assert high == pytest.approx(10.0 / np.sqrt(2), rel=1e-3), "the 2 Hz part is below the bands"
    assert bands[0] == pytest.approx(10.0 / np.sqrt(2), rel=1e-3) and max(bands[1:]) < 0.01


def test_the_dl_expected_profile_is_the_spline_on_the_stance_phase():
    from myoassist_utils.exo_ctrl import FourPointSpline
    from tools.rollout_controllers import dl_expected_profile

    params = types.SimpleNamespace(
        rise_fraction=0.278, peak_fraction=0.543, fall_fraction=0.641, peak_torque=25.0, bias_torque=0.0,
        peak_hold_time=0.0, reel_in_time=0.157,
    )  # fmt: skip
    spline = FourPointSpline(**{k: v for k, v in vars(params).items() if k != "reel_in_time"})
    phase = np.array([0.1, 0.3, 0.55, 0.59, 0.61])
    torque = dl_expected_profile(phase, params, stride=1.1, stance_fraction=0.6)
    assert torque[0] == 0.0, "reel-in: 0.157 s of 1.1 s"
    np.testing.assert_allclose(torque[1:4], [spline.torque(0.6 * p / 0.6) for p in phase[1:4]])
    assert torque[4] == 0.0, "after toe-off"


def _dl_episode(case, *, hs_offset=0.010, phase_bias=0.02, blip=False, seconds=12.0):
    """A synthetic DL record: 1.1 s strides with 0.66 s contacts from 0.3 s, ticks at 175 Hz in 1200 Hz physics; the
    network's is_stance is the contact ``hs_offset`` late, its stance phase the true one + ``phase_bias``; ``blip`` puts a
    50 ms swing inside one stance."""
    from myoassist_utils.exo_ctrl import TickSchedule
    from tools.rollout_controllers import DL_EXTRA_KEYS, SIDES, Episode, Recorder

    fields = Recorder.FIELDS + tuple(f"{key}_{side}" for key in DL_EXTRA_KEYS for side in SIDES)
    t = np.arange(0.0, seconds, 1 / RATE)
    schedule = TickSchedule(rate_hz=175.0, physics_rate_hz=RATE)
    s = {name: np.zeros(len(t)) for name in fields}
    s["t"], s["tick"] = t, np.array([schedule.tick() for _ in t], dtype=float)
    since, late = (t - 0.3) % 1.1, (t - 0.3 - hs_offset) % 1.1
    contact = (t >= 0.3) & (since < 0.66)
    on = (t >= 0.3 + hs_offset) & (late < 0.66)
    if blip:
        on[(t >= 5.0) & (t < 5.05)] = False
    for side in SIDES:
        s[f"grf_{side}"] = np.where(contact, 800.0, 0.0)
        s[f"is_stance_{side}"] = on.astype(float)
        s[f"stance_phase_head_{side}"] = np.where(contact, since / 0.66 + phase_bias, 0.0)
        s[f"speed_{side}"] = np.full(len(t), 1.05)
        s[f"speed_head_{side}"] = np.full(len(t), 1.07)
        s[f"assist_on_{side}"] = (t >= 3.0).astype(float)
        s[f"gyro_z_{side}"] = 100.0 * np.sin(2 * np.pi * t / 1.1)
        reel_in = (t >= 3.0) & on & (late < 0.157)
        stance = (t >= 3.0) & on & (late >= 0.157)
        s[f"control_state_{side}"] = np.where(reel_in, 3.0, np.where(stance, 4.0, 2.0))
        held = np.maximum.accumulate(np.where(s["tick"] == 1, np.arange(len(t)), 0))  # the last tick's, between ticks
        s[f"cmd_{side}"] = s[f"applied_{side}"] = np.where(stance, 10.0, 0.0)[held]
    step_t = np.arange(1, int(seconds * 30) + 1) / 30
    steps = {"t": step_t, "pelvis_tx": 1.1 * step_t, "pelvis_ty": np.full(len(step_t), 0.9)}
    steps.update(ankle_angle_r=np.zeros(len(step_t)), ankle_angle_l=np.zeros(len(step_t)))
    return Episode(case, 0, "time limit", 1 / 30, steps, s)


def _dl_params():
    return types.SimpleNamespace(
        grf_on_newtons=100.0, grf_off_newtons=25.0, min_unload_time=0.05, min_stride_duration=0.6,
        rise_fraction=0.278, peak_fraction=0.543, fall_fraction=0.641, peak_torque=25.0, bias_torque=0.0,
        peak_hold_time=0.0, reel_in_time=0.157, dl_speed_on=0.7, dl_speed_off=0.5,
    )  # fmt: skip


def test_the_dl_analysis_measures_a_known_network():
    from tools.rollout_controllers import DL_ASSIST, DL_SHADOW, EXO_OFF, analyze_dl, report_dl

    results = {case: [_dl_episode(case)] for case in (EXO_OFF, DL_SHADOW, DL_ASSIST)}
    a = analyze_dl(results, _dl_params(), None, 175.0)
    tick = 1 / 175
    for side in ("r", "l"):
        leg = a["legs"][side]
        # 0.02 but where clipping at 1 shortens it, over the last 2% of each stance
        assert np.sqrt(np.mean(np.square(leg["phase_err"]))) == pytest.approx(0.02, abs=1e-3)
        assert leg["missed_hs"] == leg["extra_hs"] == leg["missed_to"] == leg["extra_to"] == 0
        assert np.all((np.array(leg["hs"]) >= 0.010 - 1e-9) & (np.array(leg["hs"]) <= 0.010 + tick + 1 / RATE + 1e-9))
        assert leg["agreement"][0] == pytest.approx(1 - 2 * 0.010 / 1.1, abs=2 * tick / 1.1)
        assert sum(leg["short_stance"]) == sum(leg["short_swing"]) == 0
        assert leg["speed"] == [pytest.approx(1.05)] and leg["pelvis_speed"] == [pytest.approx(1.1)]
        assert leg["assist_on_at"][0] == pytest.approx(3.0, abs=tick) and leg["assist_off"] == 0
        assert leg["tick_gaps"] == {6, 7} and leg["off_tick"] == 0 and leg["applied_error"] == 0.0
        assert np.mean(leg["reel_in"]) == pytest.approx(0.157, abs=2 / RATE)
        assert leg["gyro_hi"][0] < 1.0, "a 0.9 Hz swing has nothing above 5 Hz"
    assert a["same_as_exo_off"] == 0.0
    text = report_dl(a, policy="p.zip", config_path="c.json", sweep=[(0, 12.0)], kept=[0], rate_hz=175.0)
    assert "| stance phase vs the true one (RMSE, gate 0.03) | 0.020 | 0.020 | pass |" in text
    assert "**FAIL**" not in text.split("## The network")[0]


def test_the_dl_analysis_counts_a_flicker_that_splits_a_stride():
    from tools.rollout_controllers import DL_SHADOW, EXO_OFF, analyze_dl, report_dl

    results = {EXO_OFF: [_dl_episode(EXO_OFF)], DL_SHADOW: [_dl_episode(DL_SHADOW, blip=True)]}
    a = analyze_dl(results, _dl_params(), None, 175.0)
    leg = a["legs"]["r"]
    assert sum(leg["short_swing"]) == 1 and leg["extra_hs"] == 1 and leg["extra_to"] == 1
    assert min(leg["loop_strides"]) < 0.6
    text = report_dl(a, policy="p.zip", config_path="c.json", sweep=[(0, 12.0)], kept=[0], rate_hz=175.0)
    assert "| no is_stance flicker (runs under 200 ms) | 0 stance, 1 swing | 0 stance, 1 swing | **FAIL** |" in text


def test_the_dl_plot_draws(tmp_path):
    from tools.rollout_controllers import DL_ASSIST, DL_SHADOW, EXO_OFF, analyze_dl, plot_dl

    results = {case: [_dl_episode(case)] for case in (EXO_OFF, DL_SHADOW, DL_ASSIST)}
    a = analyze_dl(results, _dl_params(), None, 175.0)
    plot_dl(results, a, _dl_params(), None, tmp_path / "dl.png")
    assert (tmp_path / "dl.png").stat().st_size > 10_000


@pytest.fixture(scope="module")
def env():
    from tools.rollout_controllers import CASES, DEFAULT_CONFIG, make_env

    env, _ = make_env(REPO_ROOT / DEFAULT_CONFIG, CASES["4PTS"])
    yield env
    env.close()


def test_an_episode_is_reproducible_from_its_start_index(env):
    from tools.rollout_controllers import Recorder, reset_at

    actions = np.random.default_rng(0).uniform(-1.0, 1.0, (15, env.action_space.shape[0]))

    def rollout(index):
        reset_at(env, index)
        qpos = []
        for action in actions:
            env.step(action)
            qpos.append(env.sim.data.qpos.copy())
        return np.array(qpos), env.device_controller.take()

    first, record = rollout(40)
    again, _ = rollout(40)
    other, _ = rollout(80)
    np.testing.assert_array_equal(first, again)
    assert not np.allclose(first[0], other[0]), "a different start index must start from a different pose"
    assert set(record) == set(Recorder.FIELDS) and len(record["t"]) == len(actions) * 40
    assert record["tick"].sum() == len(actions) * 5, "150 Hz in 30 Hz control steps: 5 ticks per step"
