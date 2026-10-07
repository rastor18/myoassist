"""The ExoBoot's real-time Butterworth, delay timer and gyro heel-strike detector, ported to a passed-in clock.

Each is checked against a transcription of the boot's own code (filters.py, util.py, gait_state_estimators.py) with
``time.perf_counter()`` replaced by the sample time -- the only change the port makes -- and against the behaviour that
code implies: a constant passes the filter unchanged, the timer fires strictly after its delay, and a heel strike comes
one filtered peak plus that delay after the swing.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest
from scipy import signal

from myoassist_utils.exo_ctrl import Butterworth, DelayTimer, GyroHeelStrikeDetector

FS = 175.0


class _BootButterworth:
    """filters.Butterworth as the boot has it."""

    def __init__(self, N, Wn, fs):
        self.sos = signal.butter(N=N, Wn=Wn / (fs / 2), btype="low", output="sos")
        self.zi = signal.sosfilt_zi(self.sos)
        self.first_value = True

    def filter(self, new_val):
        if self.first_value:
            self.zi = self.zi * new_val
            self.first_value = False
        filtered_val, self.zi = signal.sosfilt(sos=self.sos, x=[new_val], zi=self.zi)
        return filtered_val[0]


def _boot_heel_strikes(t, gyro, *, height=100.0, n=2, wn=3.0, fs=FS, delay=0.05):
    """gait_state_estimators.GyroHeelStrikeDetector.detect on each sample, util.DelayTimer reading the sample time."""
    gyro_filter = _BootButterworth(n, wn, fs)
    gyro_history = deque([0, 0, 0], maxlen=3)
    start_time = None
    out = []
    for now, g in zip(t, gyro):
        gyro_history.appendleft(gyro_filter.filter(g))
        if gyro_history[1] > height and gyro_history[1] > gyro_history[0] and gyro_history[1] > gyro_history[2]:
            start_time = now
        if start_time is not None and now > start_time + delay:
            start_time = None
            out.append(True)
        else:
            out.append(False)
    return np.array(out)


def _walking_gyro(duration=20.0, stride=1.1, seed=0):
    """Shank gyro_z shaped like walking: a large positive swing peak and a smaller negative stance dip per stride."""
    t = np.arange(0.0, duration, 1 / FS)
    phase = (t % stride) / stride
    rng = np.random.default_rng(seed)
    swing = 320.0 * np.exp(-0.5 * ((phase - 0.78) / 0.07) ** 2)
    stance = -90.0 * np.exp(-0.5 * ((phase - 0.35) / 0.15) ** 2)
    return t, swing + stance + rng.normal(0.0, 8.0, t.size)


def test_butterworth_matches_the_boots_filter():
    t, x = _walking_gyro()
    port, boot = (
        Butterworth(order=2, cutoff_hz=3.0, fs_hz=FS),
        _BootButterworth(2, 3.0, FS),
    )
    got = np.array([port.filter(v) for v in x])
    want = np.array([boot.filter(v) for v in x])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_butterworth_starts_at_the_first_sample_and_restarts_on_reset():
    f = Butterworth(order=2, cutoff_hz=3.0, fs_hz=FS)
    assert [f.filter(42.0) for _ in range(5)] == pytest.approx([42.0] * 5, abs=1e-12)
    f.reset()
    assert f.filter(-7.0) == pytest.approx(-7.0, abs=1e-12)


def test_butterworth_cutoff_must_be_below_nyquist():
    with pytest.raises(ValueError, match="cutoff"):
        Butterworth(order=2, cutoff_hz=90.0, fs_hz=FS)


def test_delay_timer_fires_strictly_after_the_delay_until_reset():
    timer = DelayTimer(0.05)
    assert not timer.check(10.0), "an unstarted timer never fires"
    timer.start(1.0)
    assert not timer.check(1.05)
    assert timer.check(1.0500001) and timer.check(2.0)
    timer.start(1.5)  # restart
    assert not timer.check(1.52)
    timer.reset()
    assert not timer.check(5.0)


def test_gyro_heel_strikes_match_the_boots_detector():
    t, gyro = _walking_gyro()
    detector = GyroHeelStrikeDetector()
    got = np.array([detector.detect(a, g) for a, g in zip(t, gyro)])
    want = _boot_heel_strikes(t, gyro)
    assert want.sum() >= 15, "the walking signal should produce a strike per stride"
    np.testing.assert_array_equal(got, want)


def test_gyro_heel_strike_comes_one_peak_sample_plus_the_delay_after_swing():
    """One clean swing peak: the strike is the first sample strictly after (the sample after the filtered peak) + delay."""
    t = np.arange(0.0, 3.0, 1 / FS)
    gyro = 300.0 * np.exp(-0.5 * ((t - 1.0) / 0.06) ** 2)
    filtered = np.array([y for y in _filtered(gyro)])
    peak = int(np.argmax(filtered))
    detector = GyroHeelStrikeDetector(delay=0.05)
    strikes = [i for i, (a, g) in enumerate(zip(t, gyro)) if detector.detect(a, g)]
    timer_start = t[peak + 1]  # the peak is only known on the next sample
    expected = int(np.argmax(t > timer_start + 0.05))
    assert strikes == [expected]


def test_gyro_peak_below_threshold_is_not_a_strike():
    t = np.arange(0.0, 3.0, 1 / FS)
    detector = GyroHeelStrikeDetector(threshold=100.0)
    assert not any(detector.detect(a, g) for a, g in zip(t, 80.0 * np.exp(-0.5 * ((t - 1.0) / 0.06) ** 2)))


def test_gyro_detector_reset_forgets_a_pending_strike():
    t = np.arange(0.0, 3.0, 1 / FS)
    gyro = 300.0 * np.exp(-0.5 * ((t - 1.0) / 0.06) ** 2)
    detector = GyroHeelStrikeDetector()
    fired = [detector.detect(a, g) for a, g in zip(t, gyro)]
    first = int(np.argmax(fired))
    detector.reset()
    # Replaying up to just before the strike, then resetting, leaves no timer running.
    for a, g in zip(t[: first - 1], gyro[: first - 1]):
        detector.detect(a, g)
    detector.reset()
    assert not any(detector.detect(a, 0.0) for a in t[first - 1 :])


def _filtered(x):
    f = Butterworth(order=2, cutoff_hz=3.0, fs_hz=FS)
    return [f.filter(v) for v in x]
