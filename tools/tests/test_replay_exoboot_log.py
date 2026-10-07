"""tools/replay_exoboot_log.py, on a synthetic boot log written the way the boot writes one.

Real session logs are human-subject data and stay out of the repo, so this builds a log with the boot's columns,
its four states, a known reel-in, and the boot's quirk of leaving commanded_torque stale outside stance -- then
checks the tool recovers what was put in. A tool that silently mis-read a column, or compared stale commands,
would report a port as faithful or broken for the wrong reason.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from myoassist_utils.exo_ctrl import (
    FourPointSpline,
    GyroHeelStrikeDetector,
    StrideAverageGaitPhaseEstimator,
)

SPLINE = dict(
    rise_fraction=0.278,
    peak_fraction=0.543,
    fall_fraction=0.641,
    peak_torque=25.0,
    bias_torque=3.0,
)
REEL_IN, REEL_OUT_TIME, STRIDE, RATE = 0.15, 0.07, 1.147, 175.0
DURATION = 60.0
GYRO_THRESHOLD = 90.0  # not the boot's default (100), so a tool that ignored CONFIG.csv would detect differently


def _times(duration=DURATION):
    return np.arange(0.0, duration, 1 / RATE) + 0.0057


def _walking_gyro(t):
    """Shank gyro_z shaped like walking: a swing peak and a stance dip per stride."""
    phase = (t % STRIDE) / STRIDE
    noise = np.random.default_rng(0).normal(0.0, 5.0, t.size)
    return 300.0 * np.exp(-0.5 * ((phase - 0.78) / 0.07) ** 2) - 80.0 * np.exp(-0.5 * ((phase - 0.35) / 0.15) ** 2) + noise


def _boot_log(duration=DURATION, *, heel_strike=None, gyro=None):
    """One leg's log as the boot would write it, with a fixed reel-in and reel-out.

    ``heel_strike`` marks the strike rows; by default a strike every ``STRIDE`` s, logged on the first row after it.
    """
    spline, estimator = FourPointSpline(**SPLINE), StrideAverageGaitPhaseEstimator()
    t = _times(duration)
    if heel_strike is None:
        strikes = np.arange(0.5, duration, STRIDE)
        heel_strike = np.array([np.any((strikes > tt - 1 / RATE) & (strikes <= tt)) for tt in t])
    else:
        strikes = t[heel_strike]
    rows, last_cmd = [], np.nan
    for i, tt in enumerate(t):
        hs = bool(heel_strike[i])
        phase = estimator.estimate(tt, hs)
        since = tt - strikes[strikes <= tt].max() if (strikes <= tt).any() else np.inf
        if phase is None:
            state = 2
        elif since < REEL_IN:
            state = 3
        elif phase <= 0.6:
            state = 4
        elif since < phase_toe_off_time(estimator) + REEL_OUT_TIME:
            state = 1
        else:
            state = 2
        if state == 4:
            last_cmd = spline.torque(phase)
        rows.append(
            dict(
                loop_time=tt,
                did_heel_strike=int(hs),
                gait_phase=np.nan if phase is None else phase,
                commanded_torque=last_cmd,  # stale outside stance, as on the boot
                ankle_torque_from_current=last_cmd if state == 4 else 0.5,
                controller={1: 1, 2: 2, 3: 3, 4: 8}[state],
                control_state=state,
                **({} if gyro is None else {"gyro_z": gyro[i]}),
            )
        )
    return pd.DataFrame(rows)


def phase_toe_off_time(estimator):
    return 0.6 * estimator.mean_stride_duration


def _config(**extra):
    return dict(
        STANCE_CONTROL_STYLE="StanceCtrlStyle.FOURPOINTSPLINE",
        SWING_ONLY=False,
        RISE_FRACTION=SPLINE["rise_fraction"],
        PEAK_FRACTION=SPLINE["peak_fraction"],
        FALL_FRACTION=SPLINE["fall_fraction"],
        PEAK_TORQUE=SPLINE["peak_torque"],
        SPLINE_BIAS=SPLINE["bias_torque"],
        TOE_OFF_FRACTION=0.6,
        NUM_STRIDES_REQUIRED=2,
        **extra,
    )


def _write_session(d, logs: dict, config: dict) -> str:
    prefix = str(d / "20990101_0000_TEST_")
    for side, log in logs.items():
        log.to_csv(prefix + f"{side}.csv", index=False)
    pd.DataFrame([{"loop_time": 0.0, **config}]).to_csv(prefix + "CONFIG.csv", index=False)
    return prefix


@pytest.fixture(scope="module")
def log_prefix(tmp_path_factory):
    return _write_session(
        tmp_path_factory.mktemp("session"),
        {side: _boot_log() for side in ("LEFT", "RIGHT")},
        _config(),
    )


@pytest.fixture(scope="module")
def gyro_log_prefix(tmp_path_factory):
    """A log whose strikes are what the boot's gyro detector, at CONFIG's threshold, finds in its gyro_z column."""
    t = _times()
    gyro = _walking_gyro(t)
    detector = GyroHeelStrikeDetector(threshold=GYRO_THRESHOLD)
    log = _boot_log(
        heel_strike=np.array([detector.detect(a, g) for a, g in zip(t, gyro)]),
        gyro=gyro,
    )
    config = _config(
        HS_GYRO_THRESHOLD=GYRO_THRESHOLD,
        HS_GYRO_FILTER_N=2,
        HS_GYRO_FILTER_WN=3.0,
        HS_GYRO_DELAY=0.05,
        TARGET_FREQ=RATE,
    )
    return _write_session(tmp_path_factory.mktemp("gyro_session"), {"LEFT": log, "RIGHT": log}, config)


def test_recovers_an_exact_port(log_prefix):
    from tools.replay_exoboot_log import (
        Log,
        check_spline,
        params_in_window,
        reel_in_durations,
        replay_controller,
        replay_phase,
        toe_off_offsets,
    )

    params = params_in_window(pd.read_csv(log_prefix + "CONFIG.csv"), 10.0, 55.0)
    log = Log.read(log_prefix + "LEFT.csv")
    window = (log.t >= 10.0) & (log.t <= 55.0)

    assert np.abs(check_spline(log, params, window)).max() < 1e-12
    phase = replay_phase(log, params)
    np.testing.assert_allclose(phase[window], log.phase[window], atol=1e-12)
    assert np.all(toe_off_offsets(log, params, 10.0, 55.0) == 0)
    reel = reel_in_durations(log, 10.0, 55.0)
    assert reel.mean() == pytest.approx(REEL_IN, abs=1 / RATE)

    port = replay_controller(log.t, log.heel_strike, params, reel_in_time=REEL_IN)
    boot = log.assist_command()
    # Stale commands outside stance must not count: the assist command is zero there.
    assert np.all(boot[window & (log.state != 4)] == 0.0)
    assert np.sqrt(np.mean((port - boot)[window] ** 2)) < 0.5


def test_refuses_a_window_with_a_parameter_change(log_prefix):
    from tools.replay_exoboot_log import params_in_window

    config = pd.read_csv(log_prefix + "CONFIG.csv")
    changed = pd.concat([config, config.assign(loop_time=30.0, PEAK_TORQUE=20.0)], ignore_index=True)
    with pytest.raises(ValueError, match="change inside the window"):
        params_in_window(changed, 10.0, 55.0)


def test_runs_end_to_end(log_prefix, capsys):
    from tools.replay_exoboot_log import main

    main([log_prefix, "--start", "10", "--end", "55"])
    out = capsys.readouterr().out
    assert "1. spline" in out and "4. controller" in out and "30 Hz" not in out
    assert "6. in-loop 150 Hz" in out and "5. gyro" not in out, "no gyro_z column, so no gyro layer"
    suggested = float(re.search(r"exo_controller_params\.reel_in_time ([0-9.]+)", out).group(1))
    assert suggested == pytest.approx(REEL_IN, abs=1 / RATE)


def test_gyro_detector_parameters_come_from_the_config(gyro_log_prefix, log_prefix):
    from tools.replay_exoboot_log import params_in_window

    params = params_in_window(pd.read_csv(gyro_log_prefix + "CONFIG.csv"), 10.0, 55.0)
    assert params.gyro["threshold"] == GYRO_THRESHOLD and params.gyro_defaulted == ()
    # A CONFIG.csv that does not record them falls back to the boot's defaults, and says so.
    params = params_in_window(pd.read_csv(log_prefix + "CONFIG.csv"), 10.0, 55.0)
    assert params.gyro["threshold"] == 100.0 and "HS_GYRO_THRESHOLD" in params.gyro_defaulted


def test_gyro_port_recovers_the_logged_strikes(gyro_log_prefix):
    from tools.replay_exoboot_log import (
        Log,
        params_in_window,
        replay_gyro_strikes,
        strike_offsets,
    )

    params = params_in_window(pd.read_csv(gyro_log_prefix + "CONFIG.csv"), 10.0, 55.0)
    log = Log.read(gyro_log_prefix + "LEFT.csv")
    window = (log.t >= 10.0) & (log.t <= 55.0)
    offsets, extra = strike_offsets(replay_gyro_strikes(log.t, log.gyro, params.gyro), log.heel_strike, window)
    assert len(offsets) >= 35 and np.all(offsets == 0) and extra == 0


def test_strike_offsets_count_shifted_missed_and_extra_strikes():
    from tools.replay_exoboot_log import strike_offsets

    logged = np.zeros(100, dtype=bool)
    logged[[10, 40, 70]] = True
    port = np.zeros(100, dtype=bool)
    port[[11, 40, 55]] = True  # one a row late, one exact, one extra; the strike at 70 is missed
    offsets, extra = strike_offsets(port, logged, np.ones(100, dtype=bool))
    assert offsets[:2].tolist() == [1, 0] and abs(offsets[2]) > 3 and extra == 1


@pytest.mark.parametrize("rate", [150.0, RATE])
def test_in_loop_pipeline_tracks_the_boot_rate_replay(gyro_log_prefix, rate):
    """On the boot's 175 Hz rows, in-loop ticks at the env's 150 Hz (or the boot's own 175) see each strike within a
    tick and hold torque for one, so timing stays within a tick and size within a percent or two."""
    from tools.replay_exoboot_log import Log, compare_in_loop, params_in_window

    params = params_in_window(pd.read_csv(gyro_log_prefix + "CONFIG.csv"), 10.0, 55.0)
    log = Log.read(gyro_log_prefix + "LEFT.csv")
    results = compare_in_loop(log, params, 10.0, 55.0, reel_in_time=REEL_IN, rate=rate)
    assert set(results) == {"logged strikes", "gyro port"}
    for name, (lag, peak, impulse) in results.items():
        assert len(lag) >= 35, name
        assert abs(lag.mean()) < 1 / rate * 1e3, f"{name}: lag {lag.mean():.2f} ms"
        assert abs(peak.mean()) < 0.01 and abs(impulse.mean()) < 0.02, name


def test_in_loop_ticks_follow_the_device_envs_schedule():
    from tools.replay_exoboot_log import in_loop_ticks

    ticks = in_loop_ticks(1.0, 175.0)
    assert len(ticks) == 176  # ticks 0..175 at k/175 s, rounded up to the 1200 Hz substep grid
    assert np.all((ticks - np.arange(176) / 175.0 >= -1e-12) & (ticks - np.arange(176) / 175.0 < 1 / 1200))
    ticks = in_loop_ticks(1.0, 150.0)
    assert np.allclose(ticks, np.arange(151) / 150.0, atol=1e-12)  # every 8th substep, exactly on time
