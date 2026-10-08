"""The ExoBoot four-point spline, ported for use as a scripted exo controller in the RL env.

Almost every way this can be wrong is silent. A spline evaluated a hair past phase 1 returns NaN, which
poisons a whole PPO batch without an error. An actuator on the wrong joint, or on the other leg, takes a
plausible-looking torque to the wrong place. A foot loaded at reset counted as a heel strike, or episode start
counted as a stride, gives a phase that runs fast for the first valid stride. None of these crash, and all of them
would change what an experiment measures.

These tests need no simulation, except the actuator tests that read real composed models.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import interpolate

from myoassist_utils.exo_ctrl import (
    ExoBootFourPointSplineController,
    FourPointSpline,
    StrideAverageGaitPhaseEstimator,
    VgrfHeelStrikeDetector,
    ankle_torque_actuator,
    torque_actuator_params,
)

# The ExoBoot's tuned hardware values (NeuMoveExoBoot/config_util.py:69-80).
EXOBOOT = dict(
    rise_fraction=0.278,
    peak_fraction=0.543,
    fall_fraction=0.641,
    peak_torque=25.0,
    bias_torque=3.0,
)
GRID = np.linspace(0.0, 1.0, 2001)
RATE_HZ = 150.0  # the controller's rate in the device env


def _controller(**kwargs) -> ExoBootFourPointSplineController:
    return ExoBootFourPointSplineController(
        spline=FourPointSpline(**EXOBOOT),
        heel_strike_detector=VgrfHeelStrikeDetector(grf_on=50.0, grf_off=25.0, min_unload_time=0.05),
        phase_estimator=StrideAverageGaitPhaseEstimator(),
        **kwargs,
    )


# --- the torque profile -------------------------------------------------------------------------------------------


def test_spline_matches_the_hardware_construction():
    """Same knots, same interpolant: the profile must equal the one the boot computes, point for point.

    The reference below is ExoBoot's _get_spline_x / _get_spline_y fed to scipy's pchip, exactly as
    GenericSplineController.update_spline does. It is reproduced rather than imported because the ExoBoot
    modules import hardware drivers (flexsea) at module level.
    """
    spline = FourPointSpline(**EXOBOOT)
    x = [
        0,
        EXOBOOT["rise_fraction"],
        EXOBOOT["peak_fraction"],
        EXOBOOT["fall_fraction"],
        1,
    ]
    b, p = EXOBOOT["bias_torque"], EXOBOOT["peak_torque"]
    reference = interpolate.pchip(x, [b, b, p, b, b], extrapolate=False)

    ours = np.array([spline.torque(phase) for phase in GRID])
    np.testing.assert_allclose(ours, reference(GRID), rtol=0, atol=1e-12)


def test_spline_shape_with_hardware_defaults():
    """Bias at both ends and before the rise, the configured peak at the configured phase, no overshoot."""
    spline = FourPointSpline(**EXOBOOT)
    torque = np.array([spline.torque(phase) for phase in GRID])

    assert spline.torque(0.0) == pytest.approx(3.0) and spline.torque(1.0) == pytest.approx(3.0)
    assert spline.torque(EXOBOOT["peak_fraction"]) == pytest.approx(25.0)
    assert GRID[torque.argmax()] == pytest.approx(EXOBOOT["peak_fraction"], abs=1e-3)
    before_rise = GRID <= EXOBOOT["rise_fraction"]
    np.testing.assert_allclose(
        torque[before_rise],
        3.0,
        atol=1e-9,
        err_msg="torque should sit at the bias before the rise",
    )
    # PCHIP is shape-preserving: nothing above the peak or below the bias, monotone on each side of the peak.
    assert torque.max() <= 25.0 + 1e-9 and torque.min() >= 3.0 - 1e-9
    rising = (GRID >= EXOBOOT["rise_fraction"]) & (GRID <= EXOBOOT["peak_fraction"])
    falling = (GRID >= EXOBOOT["peak_fraction"]) & (GRID <= EXOBOOT["fall_fraction"])
    assert np.all(np.diff(torque[rising]) >= -1e-12) and np.all(np.diff(torque[falling]) <= 1e-12)


@pytest.mark.parametrize("phase", [0.0, 1.0, 1.0 + 1e-12, -1e-12, 1.5, -0.5])
def test_spline_never_returns_nan(phase):
    """extrapolate=False gives NaN a single ulp outside [0, 1], and one NaN in the action wrecks a PPO batch."""
    torque = FourPointSpline(**EXOBOOT).torque(phase)
    assert math.isfinite(torque), f"torque at phase {phase!r} is {torque}; the phase clamp is missing"


def test_no_phase_means_no_torque():
    """Gait not steady yet -> zero, not the bias: the hardware commands 0 when gait_phase is None."""
    assert FourPointSpline(**EXOBOOT).torque(None) == 0.0


def test_peak_hold_holds_the_peak():
    spline = FourPointSpline(**{**EXOBOOT, "fall_fraction": 0.75}, peak_hold_time=0.1)
    for phase in np.linspace(0.543, 0.643, 11):
        assert spline.torque(phase) == pytest.approx(25.0, abs=1e-9)


@pytest.mark.parametrize(
    "overrides",
    [
        {"rise_fraction": 0.6},  # rise after peak
        {"peak_fraction": 0.7},  # peak after fall
        {"fall_fraction": 1.0},  # fall at the cycle end
        {"rise_fraction": 0.0},  # rise at the cycle start
        {"peak_hold_time": 0.2},  # hold runs past fall
        {"peak_torque": -5.0},  # the ankle exos cannot dorsiflex
        {"bias_torque": math.nan},
    ],
)
def test_spline_rejects_parameters_it_cannot_honor(overrides):
    """Fail at construction, naming the parameter, rather than mid-episode or with a scipy knot error."""
    with pytest.raises(ValueError):
        FourPointSpline(**{**EXOBOOT, **overrides})


# --- the ankle actuators ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["DephyExoBoot_L1", "Tutorial_L1"])
def test_ankle_actuators_are_read_from_the_composed_models(device):
    """Read from the model, not hardcoded -- and the two legs must map identically (signs are not mirrored)."""
    import mujoco

    from myoassist_utils.compose import compose_env_model

    model = mujoco.MjModel.from_xml_string(compose_env_model("myolegs22", device))
    for name, side in (("Exo_R", "r"), ("Exo_L", "l")):
        actuator_id = ankle_torque_actuator(model, name, side)
        assert torque_actuator_params(model, actuator_id) == (100.0, 1.0, -1.0, 0.0), name


@pytest.mark.parametrize(
    "device, actuator, side, why",
    [
        (
            "UTAnkleExo_L2",
            "UTAnkleExo_L2_part2part3act_dx",
            "r",
            "tendon drive with filter dynamics",
        ),
        (
            "STRIDE_L2",
            "STRIDE_L2_cable_r",
            "r",
            "tendon drive in newtons, ctrlrange [0, 400]",
        ),
        (
            "Hippo_L1",
            "Exo_R",
            "r",
            "a hip exo: passes every actuator check, but it is not an ankle",
        ),
        (
            "DephyExoBoot_L1",
            "Exo_L",
            "r",
            "the left ankle's actuator named for the right leg",
        ),
    ],
)
def test_actuators_that_would_be_driven_wrongly_are_refused(device, actuator, side, why):
    """Each of these would get a plausible torque in the wrong place without any error, so they are refused."""
    import mujoco

    from myoassist_utils.compose import compose_env_model

    model = mujoco.MjModel.from_xml_string(compose_env_model("myolegs22", device))
    with pytest.raises(ValueError):
        ankle_torque_actuator(model, actuator, side)


# --- heel strikes and phase ---------------------------------------------------------------------------------------


def _ticks(rate, duration, grf):
    """Sample grf(t) once per controller tick, as the env does, and return the strike times."""
    detector = VgrfHeelStrikeDetector(grf_on=50.0, grf_off=25.0, min_unload_time=0.05)
    ts = np.arange(0.0, duration, 1 / rate)
    return [float(t) for t in ts if detector.detect(t, grf(t))]


def test_a_foot_loaded_at_reset_is_not_a_heel_strike():
    """An episode that starts mid-stance must not strike on its first tick: a strike needs an observed unload."""
    # Loaded at reset, toe-off at 0.4 s, genuine heel strike at 0.9 s.
    strikes = _ticks(RATE_HZ, 1.2, lambda t: 0.0 if 0.4 <= t < 0.9 else 400.0)
    assert strikes and strikes[0] >= 0.9, f"a foot already loaded at reset was counted as a strike: {strikes}"
    assert len(strikes) == 1 and strikes[0] == pytest.approx(0.9, abs=1 / RATE_HZ)


def _bouncy_stance(t):
    """Swing until 0.1 s, then stance with two brief unloads: one right after the strike, one deep in stance."""
    if t < 0.1 or t >= 0.7:
        return 0.0
    if 0.12 <= t < 0.135 or 0.40 <= t < 0.42:
        return 5.0
    return 400.0


@pytest.mark.parametrize("rate", [RATE_HZ, 1200])
def test_one_strike_per_stance_despite_bounces(rate):
    """A brief unload must not split the stride, whether it comes just after the strike or late in stance.

    The late one is the case a dwell measured from strike onset misses. A split stride is too short to pass the
    stride bounds, so it knocks phase to None and the exo cuts out for two strides. Both rates see both unloads
    (15 and 20 ms): the env's 150 Hz, and every 1200 Hz physics substep.
    """
    strikes = _ticks(rate, 1.0, _bouncy_stance)
    assert len(strikes) == 1 and strikes[0] == pytest.approx(0.1, abs=1 / rate), f"expected one strike at 0.1 s, got {strikes}"


def test_a_sustained_unload_is_a_toe_off():
    """The debounce must still let a real swing through, or no second strike is ever seen."""
    strikes = _ticks(RATE_HZ, 3.0, lambda t: 0.0 if (t % 1.1) / 1.1 >= 0.6 else 400.0)
    assert len(strikes) == 2 and strikes == pytest.approx([1.1, 2.2], abs=1 / RATE_HZ), strikes


def test_hysteresis_band_does_not_toggle():
    """A signal wandering inside (grf_off, grf_on) neither strikes nor releases."""
    detector = VgrfHeelStrikeDetector(grf_on=50.0, grf_off=25.0, min_unload_time=0.0)
    detector.detect(0.0, 0.0)
    assert not any(detector.detect(0.01 * k, grf) for k, grf in enumerate([30, 45, 26, 49, 40], start=1))
    assert detector.detect(0.1, 60.0) is True
    assert not any(detector.detect(0.1 + 0.01 * k, grf) for k, grf in enumerate([30, 45, 26, 49], start=1))


def _scuffing_gait(t):
    """Stance from each strike at k * 1.1 s for 0.66 s, and a 20 ms, 500 N scuff 0.25 s into every swing."""
    since = t % 1.1
    if since < 0.66:
        return 400.0
    return 500.0 if 0.91 <= since < 0.93 else 0.0


@pytest.mark.parametrize("rate", [RATE_HZ, 1200])
def test_a_brief_touch_in_swing_is_not_a_heel_strike(rate):
    """A foot that scuffs the ground in swing loads it for milliseconds, at forces real strikes also reach. Counted,
    it splits the stride and the exo cuts out for two strides; with min_contact_time it is not counted."""
    ts = np.arange(0.0, 4.0, 1 / rate)
    debounced = VgrfHeelStrikeDetector(grf_on=100.0, grf_off=25.0, min_unload_time=0.05, min_contact_time=0.05)
    hits = [t for t in ts if debounced.detect(t, _scuffing_gait(t))]
    assert len(hits) == 3, f"expected the strikes at 1.1, 2.2 and 3.3 s only, got {hits}"
    plain = VgrfHeelStrikeDetector(grf_on=100.0, grf_off=25.0, min_unload_time=0.05)
    assert sum(plain.detect(t, _scuffing_gait(t)) for t in ts) == 6, "without it each scuff is a strike"


def test_a_confirmed_strike_is_dated_to_when_the_contact_began():
    detector = VgrfHeelStrikeDetector(grf_on=100.0, grf_off=25.0, min_unload_time=0.05, min_contact_time=0.05)
    ts = np.arange(0.0, 2.0, 1 / RATE_HZ)
    hits = [t for t in ts if detector.detect(t, 0.0 if t < 1.0 else 400.0)]
    onset = ts[np.searchsorted(ts, 1.0)]
    assert len(hits) == 1 and hits[0] == pytest.approx(onset + 0.05, abs=1 / RATE_HZ)
    assert detector.strike_time == onset, "reported once confirmed, but dated to the first loaded tick"


def test_without_min_contact_time_a_strike_is_reported_on_its_first_tick():
    """min_contact_time 0 is the detector as it was: the gyro port and the boot's own strikes date a strike to the tick
    that detects it, and so does this."""
    detector = VgrfHeelStrikeDetector(grf_on=100.0, grf_off=25.0, min_unload_time=0.05)
    detector.detect(0.0, 0.0)
    assert detector.detect(0.5, 400.0) is True and detector.strike_time == 0.5


def test_the_contact_debounce_moves_nothing_but_the_tick_that_reports_the_strike():
    """Dated to the onset, a confirmed strike gives the same stride, phase and torque as one reported at once, from
    the tick it is reported on. Before that the controller is still in the last stride's swing, past toe-off."""
    reported_at_once = _controller(reel_in_time=0.157)
    confirmed = ExoBootFourPointSplineController(
        spline=FourPointSpline(**EXOBOOT),
        heel_strike_detector=VgrfHeelStrikeDetector(grf_on=50.0, grf_off=25.0, min_unload_time=0.05, min_contact_time=0.05),
        phase_estimator=StrideAverageGaitPhaseEstimator(),
        reel_in_time=0.157,
    )
    ticks, at_once, strikes = _walk(reported_at_once, RATE_HZ, np.full(8, 1.1))
    _, debounced, _ = _walk(confirmed, RATE_HZ, np.full(8, 1.1))
    k = np.searchsorted(strikes, ticks, side="right") - 1
    settled = (ticks - strikes[np.maximum(k, 0)]) > 0.05 + 1 / RATE_HZ
    assert at_once.max() > 20.0, "the comparison must include assisted strides"
    np.testing.assert_allclose(debounced[settled], at_once[settled], atol=1e-12)
    assert np.all(debounced[~settled] == 0.0) and np.all(at_once[~settled] == 0.0), "reel-in covers the debounce"


def test_diagnostics_date_each_reported_strike_and_only_on_its_tick():
    """``strike_time`` is the detector's dating of the strike the tick reports (the contact's onset), and -1 on every
    other tick, so a record of the diagnostics alone says when each strike happened."""
    controller = ExoBootFourPointSplineController(
        spline=FourPointSpline(**EXOBOOT),
        heel_strike_detector=VgrfHeelStrikeDetector(grf_on=100.0, grf_off=25.0, min_unload_time=0.05, min_contact_time=0.05),
        phase_estimator=StrideAverageGaitPhaseEstimator(),
    )
    ts = np.arange(0.0, 2.0, 1 / RATE_HZ)
    rows = []
    for t in ts:
        controller.step(float(t), 0.0 if t < 1.0 else 400.0)
        rows.append(controller.diagnostics())
    reported = [d for d in rows if d["heel_strike"] == 1.0]
    assert len(reported) == 1 and reported[0]["strike_time"] == ts[np.searchsorted(ts, 1.0)]
    assert all(d["strike_time"] == -1.0 for d in rows if d["heel_strike"] == 0.0)


def test_a_scuff_does_not_cost_the_phase():
    """The regression: on a gait that scuffs in every swing, phase stays valid once two strides are in."""
    controller = ExoBootFourPointSplineController(
        spline=FourPointSpline(**EXOBOOT),
        heel_strike_detector=VgrfHeelStrikeDetector(grf_on=100.0, grf_off=25.0, min_unload_time=0.05, min_contact_time=0.05),
        phase_estimator=StrideAverageGaitPhaseEstimator(),
    )
    valid = []
    for t in np.arange(0.0, 8.0, 1 / RATE_HZ):
        controller.step(float(t), _scuffing_gait(t))
        valid.append((t, controller.diagnostics()["phase_valid"]))
    assert all(v == 1.0 for t, v in valid if t > 3.3 + 1 / RATE_HZ + 0.05), "a scuff knocked the phase out"


def test_the_gyro_detector_dates_a_strike_to_the_tick_that_detects_it():
    from myoassist_utils.exo_ctrl import GyroHeelStrikeDetector

    detector = GyroHeelStrikeDetector()
    ts = np.arange(0.0, 3.0, 1 / 175.0)
    swing_peaks = 300.0 * np.exp(-0.5 * (((ts % 1.1) - 0.8) / 0.07) ** 2)
    hits = [t for t, g in zip(ts, swing_peaks) if detector.detect(t, g)]
    assert hits and detector.strike_time == hits[-1]


def _strikes_every(period, count, first=0.0):
    return [first + k * period for k in range(count)]


def test_phase_needs_two_genuine_strides_then_ramps():
    """None until the third strike (the first stride is always rejected), then phase = time since strike / stride."""
    estimator = StrideAverageGaitPhaseEstimator()
    strikes = _strikes_every(1.1, 5, first=0.5)
    phases = {}
    for t in np.round(np.arange(0.5, 5.5, 0.05), 10):
        phases[t] = estimator.estimate(t, did_heel_strike=any(abs(t - s) < 1e-9 for s in strikes))

    assert all(v is None for t, v in phases.items() if t < strikes[2]), "phase became valid before two whole strides"
    assert phases[round(strikes[2] + 0.55, 10)] == pytest.approx(0.55 / 1.1)


def test_episode_start_is_not_a_stride():
    """Sim time starts at 0. Seeding the last strike at 0 would count 0 -> first strike as a stride.

    Here that interval (0.9 s) is inside the stride bounds, so with a 0 seed the second strike would already make
    phase valid, with a stride average of (0.9 + 1.1) / 2 that runs phase fast for the whole first stride.
    """
    estimator = StrideAverageGaitPhaseEstimator()
    estimator.estimate(0.9, did_heel_strike=True)
    assert estimator.estimate(2.0, did_heel_strike=True) is None, "the interval from episode start was used as a stride"
    assert estimator.estimate(3.1, did_heel_strike=True) == pytest.approx(0.0)
    assert estimator.mean_stride_duration == pytest.approx(1.1)


def test_phase_times_out_without_a_strike():
    """A stride running past 1.2 x max_stride_duration (2.4 s by default) invalidates phase, as on hardware."""
    estimator = StrideAverageGaitPhaseEstimator()
    for t in _strikes_every(1.1, 3):
        estimator.estimate(t, did_heel_strike=True)
    assert estimator.estimate(2.2 + 2.3, did_heel_strike=False) is not None
    assert estimator.estimate(2.2 + 2.5, did_heel_strike=False) is None


@pytest.mark.parametrize("bad_stride", [0.4, 2.5])
def test_an_out_of_bounds_stride_invalidates_phase(bad_stride):
    estimator = StrideAverageGaitPhaseEstimator()
    t = 0.0
    for stride in (1.1, 1.1, 1.1, bad_stride):
        estimator.estimate(t, did_heel_strike=True)
        t += stride
    assert estimator.estimate(t, did_heel_strike=True) is None


def test_phase_is_clipped_to_one():
    """A stride longer than the average holds phase at 1 (torque at the bias) rather than running past it."""
    estimator = StrideAverageGaitPhaseEstimator()
    for t in _strikes_every(1.0, 3):
        estimator.estimate(t, did_heel_strike=True)
    assert estimator.estimate(2.0 + 1.3, did_heel_strike=False) == 1.0


# --- the assembled controller -------------------------------------------------------------------------------------


def _walk(controller, rate, stride_durations, t0=0.137, duty=0.6):
    """Drive the controller with a square-wave GRF at `rate`. Returns tick times, torques, and true strike times."""
    strikes = t0 + np.concatenate([[0.0], np.cumsum(stride_durations)])

    def grf(t):
        k = np.searchsorted(strikes, t, side="right") - 1
        if k < 0 or k >= len(stride_durations):
            return 0.0
        return 400.0 if t - strikes[k] < duty * stride_durations[k] else 0.0

    ticks = np.arange(0.1, strikes[-2], 1 / rate)
    torques = np.array([controller.step(t, grf(t)) for t in ticks])
    return ticks, torques, strikes


def test_controller_warms_up_like_the_hardware_then_assists():
    """Start mid-stance: no torque until the third genuine strike, then the stance profile every stride."""
    controller = _controller()
    ticks, torques, strikes = _walk(controller, rate=RATE_HZ, stride_durations=np.full(8, 1.1))

    first_assist = ticks[np.argmax(torques > 0)]
    assert strikes[2] <= first_assist < strikes[2] + 1 / RATE_HZ + 1e-9, (
        f"first torque at {first_assist:.3f} s; expected at the third genuine strike ({strikes[2]:.3f} s)"
    )
    assert torques[ticks > strikes[4]].max() == pytest.approx(25.0, rel=0.02)


def _phase_of(ticks, strikes):
    k = np.searchsorted(strikes, ticks, side="right") - 1
    return (ticks - strikes[k]) / np.diff(strikes)[np.minimum(k, len(strikes) - 2)]


def test_torque_is_cut_at_toe_off_and_zero_through_swing():
    """The boot hands the spline over to reel-out when phase passes TOE_OFF_FRACTION, so assist stops there.

    With the tuned knots that cut comes mid-fall -- fall_fraction (0.641) is after toe-off (0.60) -- so torque drops
    from ~11 N*m straight to zero, and the bias is never applied in swing. Applying the spline all cycle instead
    over-delivers the boot's impulse by 23%.
    """
    controller = _controller()  # at 1200 Hz the profile lands on the true phase, so it can be checked phase by phase
    ticks, torques, strikes = _walk(controller, rate=1200, stride_durations=np.full(8, 1.1))
    steady = ticks > strikes[4]
    phase, torque = _phase_of(ticks[steady], strikes), torques[steady]

    stance, swing = (
        phase < 0.595,
        phase > 0.605,
    )  # clear of the tick on either side of the cut
    assert np.all(torque[swing] == 0.0), "torque applied after toe-off; the boot applies none in swing"
    assert torque[stance].min() == pytest.approx(3.0, abs=1e-6), "stance should never drop below the bias"
    just_before = torque[(phase > 0.59) & (phase < 0.595)]
    assert just_before.size and just_before.min() > 10.0, f"expected ~11 N*m just before toe-off, got {just_before}"


def test_toe_off_fraction_of_one_applies_the_spline_all_cycle():
    """The ungated profile is still available: toe_off_fraction = 1 applies the bias through swing."""
    controller = _controller(toe_off_fraction=1.0)
    ticks, torques, strikes = _walk(controller, rate=1200, stride_durations=np.full(8, 1.1))
    steady = ticks > strikes[4]
    phase, torque = _phase_of(ticks[steady], strikes), torques[steady]
    np.testing.assert_allclose(torque[(phase > 0.7) & (phase < 0.95)], 3.0, atol=1e-6)


def test_no_torque_during_reel_in():
    """The boot reels in its cable before stance control starts, commanding no spline torque for ~150 ms."""
    controller = _controller(reel_in_time=0.15)
    ticks, torques, strikes = _walk(controller, rate=1200, stride_durations=np.full(8, 1.1))
    steady = ticks > strikes[4]
    k = np.searchsorted(strikes, ticks[steady], side="right") - 1
    since, torque = ticks[steady] - strikes[k], torques[steady]
    assert np.all(torque[since < 0.149] == 0.0), "torque applied during reel-in"
    np.testing.assert_allclose(torque[(since > 0.151) & (since < 0.25)], 3.0, atol=1e-9)


@pytest.mark.parametrize("stride", [0.95, 1.1, 1.45])
def test_reel_in_is_a_time_not_a_phase(stride):
    """Across the validation log's speed changes reel-in did not follow step frequency, so it must end at the same time
    after the strike for short and long strides alike -- not at the same phase."""
    controller = _controller(reel_in_time=0.15)
    ticks, torques, strikes = _walk(controller, rate=1200, stride_durations=np.full(8, stride))
    steady = ticks > strikes[4]
    k = np.searchsorted(strikes, ticks[steady], side="right") - 1
    since = ticks[steady] - strikes[k]
    onset = since[(torques[steady] > 0) & (since < 0.5)].min()
    assert onset == pytest.approx(0.15, abs=2 / 1200), f"stance started {onset * 1e3:.1f} ms after the strike"


def test_negative_reel_in_is_rejected():
    with pytest.raises(ValueError, match="reel_in_time"):
        _controller(reel_in_time=-0.1)


def test_toe_off_before_the_rise_is_rejected():
    """Stance ending before torque starts rising would mean a controller that never assists."""
    with pytest.raises(ValueError, match="toe_off_fraction"):
        _controller(toe_off_fraction=0.2)


def test_diagnostics_stay_finite_through_warmup():
    """Diagnostics can be summed into training logs, so an invalid phase is reported as -1, never NaN."""
    controller = _controller()
    controller.step(0.0, 0.0)
    diagnostics = controller.diagnostics()
    assert diagnostics["phase_valid"] == 0.0 and diagnostics["phase"] == -1.0
    assert all(math.isfinite(v) for v in diagnostics.values())


def test_control_state_is_the_boots():
    """Swing until a phase exists -- the boot only enters reel-in at a heel strike once it has one -- then each stride
    reel-in, stance and, from toe-off to the next strike, swing; with the stride estimate the mean of the last two."""
    from myoassist_utils.exo_ctrl.boot_state import REEL_IN, STANCE, SWING

    controller = _controller(reel_in_time=0.15)
    strides = np.array([1.0, 1.2, 1.1, 1.1, 1.1, 1.1, 1.1])
    strikes = 0.137 + np.concatenate([[0.0], np.cumsum(strides)])
    states, estimates, ticks = [], [], np.arange(0.1, strikes[-2], 1 / RATE_HZ)
    for t in ticks:
        k = np.searchsorted(strikes, t, side="right") - 1
        controller.step(float(t), 400.0 if k >= 0 and t - strikes[k] < 0.6 * strides[min(k, len(strides) - 1)] else 0.0)
        states.append(controller.diagnostics()["control_state"])
        estimates.append(controller.diagnostics()["stride_estimate"])
    states, estimates = np.array(states), np.array(estimates)
    assert np.all(states[ticks < strikes[2]] == SWING), "warm-up, no phase yet: swing"
    k = np.searchsorted(strikes, ticks, side="right") - 1
    since = ticks - strikes[np.maximum(k, 0)]
    steady = ticks > strikes[2] + 1 / RATE_HZ
    assert np.all(states[steady & (since < 0.149)] == REEL_IN)
    assert np.all(states[steady & (since > 0.151) & (since < 0.6 * 1.1 - 0.01)] == STANCE)
    assert np.all(states[steady & (since > 0.6 * 1.2 + 0.01)] == SWING)
    # Right after the third strike the estimate is the mean of the first two whole strides, 1.0 and 1.2 s.
    after_third = (ticks > strikes[2] + 1 / RATE_HZ) & (ticks < strikes[3])
    np.testing.assert_allclose(estimates[after_third], 1.1, atol=1 / RATE_HZ)


def test_at_150_hz_the_profile_lands_within_a_tick():
    """The env's rate: each strike is seen on the first tick after it and torque is held per tick, which shifts the
    delivered profile by about one tick (6.7 ms) against the ideal one, with the peak intact."""
    rate = RATE_HZ
    strides = np.random.default_rng(0).normal(1.1, 0.03, 26)
    controller = _controller()
    ticks, torques, strikes = _walk(controller, rate, strides)
    spline, toe_off = controller.spline, controller.toe_off_fraction
    lags, peaks = [], []
    for k in range(5, len(strides) - 2):
        a, b = strikes[k], strikes[k + 1]
        tf = np.arange(a, b, 1 / 4000)
        delivered = torques[np.searchsorted(ticks, tf, side="right") - 1]  # zero-order hold, as the motor holds it
        phase = (tf - a) / (b - a)
        ideal = np.array([spline.torque(p) if p <= toe_off else 0.0 for p in phase])
        lags.append((np.sum(delivered * tf) / delivered.sum() - np.sum(ideal * tf) / ideal.sum()) * 1000)
        peaks.append((delivered.max() - ideal.max()) / ideal.max())
    assert 0.0 < np.mean(lags) < 1.5e3 / rate, f"mean lag {np.mean(lags):+.1f} ms at {rate:g} Hz"
    assert abs(np.mean(peaks)) < 0.01, f"peak torque off by {np.mean(peaks):+.1%}"
