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
