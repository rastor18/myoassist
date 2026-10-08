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


def test_vnmc_stances_are_timed_counted_through_swing_and_placed_on_the_true_stride():
    """A stance that holds through the next contact onset is one that ran through swing; each stance's end is placed on
    the stride it started in."""
    from tools.rollout_controllers import runs_of, vnmc_stance_table

    t = np.arange(0.0, 6.0, 1 / RATE)
    onsets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    state = np.full(len(t), 2.0)
    state[(t >= 1.15) & (t < 1.75)] = 4.0  # an ordinary stance, ending at phase 0.75
    state[(t >= 2.15) & (t < 3.40)] = 4.0  # one that ran through swing, past the strike at 3.0
    state[t >= 5.15] = 4.0  # cut off by the end of the record: not complete
    runs = runs_of(t, state, 4.0)
    assert len(runs) == 3 and np.isnan(runs[-1][1])
    table = vnmc_stance_table(t, state, onsets)
    np.testing.assert_allclose(table["duration"], [0.6, 1.25], atol=1 / RATE)
    assert table["through"].tolist() == [0, 1]
    np.testing.assert_allclose(table["end_phase"], [0.75, 1.40], atol=1 / RATE)


def test_the_vnmc_suite_runs_the_shadow_first_and_records_the_vnmcs_diagnostics():
    from myoassist_utils.exo_ctrl.vnmc import VNMC_DIAGNOSTICS
    from tools.rollout_controllers import SUITES, VNMC_LOG_RANGES, VNMC_RANGE_KEYS

    suite = SUITES["VNMC"]
    assert list(suite.cases) == ["VNMC shadow", "VNMC"]
    assert suite.cases["VNMC shadow"] == dict(device_controller="exoboot_vnmc", shadow_mode=True)
    assert set(VNMC_DIAGNOSTICS) | {"torque_nm"} == set(suite.extra_keys)
    for side in ("r", "l"):
        assert set(VNMC_LOG_RANGES[side]) == set(VNMC_RANGE_KEYS)


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
