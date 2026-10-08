"""The ExoBoot VNMC port (myoassist_utils/exo_ctrl/vnmc.py): its muscle, its toe-off rule, its scaling, and its path
through the boot's state machine, each on inputs whose answer is known.

The muscle steps are written out from Geyer's equations here, independently of the port; the toe-off rule and the
scaling get torque sequences chosen to hit each of their conditions; the state machine is driven tick by tick with a
muscle whose torque is scripted. Replaying the boot's own logs is tools/replay_vnmc_session.py's job (and its tests').
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from myoassist_utils.exo_ctrl import BootStateMachine, StrideAverageGaitPhaseEstimator
from myoassist_utils.exo_ctrl.boot_state import REEL_IN, REEL_OUT, STANCE, SWING
from myoassist_utils.exo_ctrl.vnmc import (
    VNMC_DIAGNOSTICS,
    MusculoTendonJoint,
    TorqueToeOffDetector,
    VNMCLeg,
    VNMCStance,
)

DT = 1 / 175

# --- the muscle -------------------------------------------------------------------------------------------------------


def test_the_first_step_from_rest():
    """At rest (angle 0, l_ce = l_opt, the series element exactly at slack) there is no force, and the contractile
    element, unloaded, shortens at v_max."""
    m = MusculoTendonJoint(timestep=DT)
    assert m.l_ce == pytest.approx(0.04) and m.activation == 0.01 and m.sensory_force == 0.0
    m.step(0.01, 0.0)
    force, length, velocity = m.states()
    assert force == 0.0 and m.torque == 0.0
    assert velocity == -1.0, "f_vce = 0 inverts to -v_max"
    assert m.l_ce == pytest.approx(0.04 - 6 * 0.04 * DT, abs=1e-15)
    assert length == pytest.approx(m.l_ce / 0.04, abs=1e-15)
    assert m.activation == 0.01, "stimulation equal to the activation changes nothing"


def test_a_stretched_step_by_hand():
    """The second step, dorsiflexed by 0.1 rad at full stimulation, from Geyer's equations written out."""
    m = MusculoTendonJoint(timestep=DT)
    m.step(0.01, 0.0)
    l_ce = m.l_ce
    m.step(1.0, -0.1)

    a = 0.01 + (1.0 - 0.01) / 0.01 * DT  # rising: tau 10 ms
    l_mtu = 0.26 + 0.04 - 0.5 * 0.04 * (-0.1)  # dorsiflexion lengthens the MTU
    f_se = ((((l_mtu - l_ce) / 0.26) - 1) / 0.1) ** 2  # stretched past slack
    f_lce = math.exp(math.log(0.05) * abs((l_ce / 0.04 - 1) / 0.56) ** 3)
    f_vce = f_se / (a * f_lce)  # no parallel or buffer force at this length
    if f_vce <= 1:
        v = (f_vce - 1) / (5 * f_vce + 1)
    elif f_vce <= 1.5:
        temp = (f_vce - 1.5) / (f_vce - 1.5 + 1)
        v = (temp + 1) / (1 - 7.56 * 5 * temp)
    else:
        v = 0.01 * (f_vce - 1.5) + 1

    assert m.activation == pytest.approx(a, rel=1e-14)
    assert m.f_mtu == pytest.approx(4000 * f_se, rel=1e-12) and m.f_mtu > 0
    assert m.torque == pytest.approx(4000 * f_se * 0.04, rel=1e-12)
    assert m.v_ce == pytest.approx(0.04 * 6 * v, rel=1e-12)
    assert m.l_ce == pytest.approx(l_ce + 0.04 * 6 * v * DT, rel=1e-12)


@pytest.mark.parametrize("stimulation, clipped", [(2.0, 1.0), (0.0, 0.01), (-1.0, 0.01)])
def test_stimulation_is_clipped_to_its_range(stimulation, clipped):
    a, b = MusculoTendonJoint(timestep=DT), MusculoTendonJoint(timestep=DT)
    a.step(stimulation, -0.2)
    b.step(clipped, -0.2)
    assert a.states() == b.states() and a.activation == b.activation


def test_activation_falls_four_times_slower_than_it_rises():
    m = MusculoTendonJoint(timestep=DT)
    m.step(1.0, 0.0)
    risen = m.activation
    m.step(0.01, 0.0)
    assert risen - m.activation == pytest.approx((risen - 0.01) / 0.04 * DT, rel=1e-12)


@pytest.mark.parametrize("rate, steps", [(175.0, 4), (150.0, 3), (1200.0, 24)])
def test_the_afferent_delay_is_20_ms_rounded_to_steps(rate, steps):
    """The force fed back is the one from round(20 ms / step) steps earlier: 4 at the boot's 175 Hz, 3 at 150 Hz."""
    m = MusculoTendonJoint(timestep=1 / rate)
    assert m.delay_steps == steps
    forces = []
    for k in range(steps + 30):
        if k >= steps:
            assert m.sensory_force == forces[k - steps], k
        else:
            assert m.sensory_force == 0.0
        m.step(0.5, -0.3)
        forces.append(m.states()[0])
    assert max(forces) > 0


def test_plantarflexion_shortens_the_muscle_and_unloads_it():
    """Held at an angle, the force settles higher the more dorsiflexed the ankle (phi < 0 lengthens the MTU)."""
    settled = {}
    for deg in (-15.0, -5.0, 5.0):
        m = MusculoTendonJoint(timestep=DT)
        for _ in range(400):
            m.step(0.3, math.radians(deg))
        settled[deg] = m.f_mtu
    assert settled[-15.0] > settled[-5.0] > settled[5.0]


def test_reset_returns_to_rest():
    m = MusculoTendonJoint(timestep=DT)
    rest = (m.states(), m.activation, m.sensory_force)
    for _ in range(20):
        m.step(0.8, -0.3)
    m.reset()
    assert (m.states(), m.activation, m.sensory_force) == rest


# --- the toe-off rule -------------------------------------------------------------------------------------------------


def _fires(torques):
    detector, peak, out = TorqueToeOffDetector(), 0.0, []
    for torque in torques:
        peak = max(peak, torque)
        out.append(detector.step(torque, peak))
    return [i for i, fired in enumerate(out) if fired]


def test_toe_off_fires_on_the_first_rise_after_four_falls_below_80_percent():
    torques = [2, 8, 14, 20, 19, 18, 17, 15.5, 15.6]  # peak 20; 4 falls latch; 15.6 rises, <= 16
    assert _fires(torques) == [8]


def test_three_falls_do_not_latch():
    assert _fires([2, 8, 14, 20, 19, 17, 15, 15.2, 14, 13, 12, 11, 11.5]) == [12], "only the second fall run latches"


def test_a_rise_above_80_percent_of_the_peak_keeps_waiting():
    """Latched, the rise at 17 (above 16) does not fire; the latch holds, and the next rise below 16 does."""
    assert _fires([2, 10, 20, 19, 18.5, 18, 17.5, 17.6, 16.5, 15.5, 15.7]) == [10]


def test_no_toe_off_while_the_peak_stays_at_5_n_m():
    """A stance whose raw peak never passes 5 N*m never toes off: its falls never count."""
    torques = [1, 3, 5, 4, 3, 2, 1, 0.5, 0.8, 0.2, 0.3] * 3
    assert _fires(torques) == []


def test_a_plateau_counts_as_falling_and_as_rising():
    """The boot's comparisons are not strict: an equal torque is both a fall (<=) and the rise (>=)."""
    assert _fires([2, 10, 20, 19, 18, 17, 15, 15]) == [7]


def test_the_latch_clears_after_firing_and_on_reset():
    detector, peak = TorqueToeOffDetector(), 0.0
    for torque in [2, 10, 20, 19, 18, 17, 15]:
        peak = max(peak, torque)
        detector.step(torque, peak)
    detector.reset()
    assert not detector.step(15.5, 20.0), "after a reset the first tick is never a fall, nor latched"


# --- reflex and scaling -----------------------------------------------------------------------------------------------


class ScriptedMuscle:
    """A muscle whose torque each step is the next in a script, and whose fed-back force is set by hand."""

    def __init__(self, torques):
        self.script = list(torques)
        self.sensory_force = 0.0
        self.steps = []  # (stimulation, angle) per step
        self.resets = 0
        self.torque = 0.0

    def reset(self):
        self.resets += 1
        self.torque = 0.0

    def step(self, stimulation, angle_rad):
        self.steps.append((stimulation, angle_rad))
        self.torque = self.script.pop(0) if self.script else 0.0

    def states(self):
        return self.torque / 160.0, 1.0, 0.0


def test_the_reflex_stimulates_with_the_delayed_force_in_stance_only():
    muscle = ScriptedMuscle([1.0, 1.0])
    stance = VNMCStance(muscle=muscle, gain=1.468, peak_torque=25.0)
    muscle.sensory_force = 0.1
    stance.stance_tick(-5.0)
    stance.idle_tick(-5.0)
    assert muscle.steps[0] == (pytest.approx(0.01 + 1.468 * 0.1, rel=1e-15), math.radians(-5.0))
    assert muscle.steps[1][0] == 0.01, "outside stance the muscle still steps, at 0.01"
    assert stance.command == 0.0


def test_scaling_as_it_ran():
    """Before any toe-off the torque is scaled by PEAK / 100; from the toe-off tick on, by PEAK / (0.8 x that stance's
    peak), so the next stance's command reaches PEAK where its raw torque reaches the last peak; clipped at PEAK."""
    first = [2.0, 10.0, 40.0, 38.0, 36.0, 34.0, 31.0, 31.5]  # peak 40, toe-off on the last tick
    second = [5.0, 20.0, 40.0, 50.0]
    stance = VNMCStance(muscle=ScriptedMuscle(first + second), gain=1.468, peak_torque=25.0)
    commands = [stance.stance_tick(0.0) for _ in first[:-1]]
    assert stance.scalefactor == 0.25 and commands == [pytest.approx(0.25 * 0.8 * x) for x in first[:-1]]
    assert stance.stance_tick(0.0) == pytest.approx(25.0 / (0.8 * 40.0) * 0.8 * 31.5) and stance.toe_off
    assert stance.scalefactor == 25.0 / (0.8 * 40.0), "the toe-off tick already scales by this stance's peak"
    stance.start_stance()
    assert stance.scalefactor == 25.0 / (0.8 * 40.0), "a new stance keeps the scaling"
    commands = [stance.stance_tick(0.0) for _ in second]
    assert commands[:3] == [pytest.approx(25.0 * x / 40.0) for x in second[:3]]
    assert commands[3] == 25.0, "clipped at the peak torque"
    assert stance.stance_peak == 50.0


def test_reset_restores_the_initial_scaling_and_the_muscle():
    muscle = ScriptedMuscle([2.0, 10.0, 40.0, 38.0, 36.0, 34.0, 31.0, 31.5])
    stance = VNMCStance(muscle=muscle, gain=1.468, peak_torque=20.0)
    for _ in range(8):
        stance.stance_tick(0.0)
    assert stance.scalefactor != 0.2
    stance.reset()
    assert stance.scalefactor == 0.2 and muscle.resets == 2 and stance.stance_peak == 0.0


@pytest.mark.parametrize("kwargs", [dict(gain=-1.0), dict(gain=float("nan")), dict(peak_torque=-1.0)])
def test_bad_parameters_are_refused(kwargs):
    with pytest.raises(ValueError):
        VNMCStance(muscle=ScriptedMuscle([]), **{"gain": 1.468, "peak_torque": 25.0, **kwargs})


# --- the leg through the boot's state machine -------------------------------------------------------------------------


class Strikes:
    """Reports the strikes it is given, dated to the tick."""

    def reset(self):
        self.strike_time = -math.inf

    def detect(self, t, signal):
        if signal:
            self.strike_time = t
        return bool(signal)


RATE = 150.0
STANCE_TORQUE = [2.0, 10.0, 20.0, 19.0, 18.0, 17.0, 15.0, 15.5]  # toe-off on its last tick


def _leg(torques, *, reel_in=0.15):
    muscle = ScriptedMuscle(torques)
    leg = VNMCLeg(
        stance=VNMCStance(muscle=muscle, gain=1.468, peak_torque=25.0),
        heel_strike_detector=Strikes(),
        phase_estimator=StrideAverageGaitPhaseEstimator(),
        state_machine=BootStateMachine(reel_in_time=reel_in, reel_out_time=0.2),
    )
    return leg, muscle


def _walk(leg, seconds, strikes, angle=-3.0):
    """Ticks at 150 Hz; a strike on the first tick at or after each time in ``strikes``. Per tick: (t, torque, state,
    diagnostics)."""
    ticks = np.arange(0.0, seconds, 1 / RATE)
    hit = set(np.searchsorted(ticks, strikes).tolist())
    return [(t, leg.step(t, (k in hit, angle)), leg.state_machine.state, leg.diagnostics()) for k, t in enumerate(ticks)]


def test_the_state_path_and_the_toe_off_one_tick_late():
    """Reel-out at start, swing, no reel-in until the phase is valid (third strike), reel-in for reel_in_time, stance
    until the VNMC's toe-off, reel-out from the next tick for 0.2 s, swing. The muscle steps on every tick and is reset
    only with the leg."""
    # Nothing scripted for the ticks outside stance would be wrong: the idle ticks consume the script too. So give the
    # scripted muscle zeros until the third strike's stance starts, then the stance profile.
    leg, muscle = _leg([])
    resets = muscle.resets
    strikes = [0.5, 1.6, 2.7]
    rows = _walk(leg, 2.7 + 0.15 + 1.0, strikes)
    states = [s for _, _, s, _ in rows]
    t = np.array([r[0] for r in rows])
    assert states[0] == REEL_OUT and states[int(0.21 * RATE)] == SWING
    assert all(s == SWING for s, tt in zip(states, t) if 0.21 < tt < 2.7), "no reel-in on a strike without a phase"
    first_reel_in = t[states.index(REEL_IN)]
    assert first_reel_in == pytest.approx(2.7, abs=1 / RATE)
    first_stance = t[states.index(STANCE)]
    assert 0.15 < first_stance - first_reel_in <= 0.15 + 1 / RATE
    assert len(muscle.steps) == len(rows) and muscle.resets == resets

    # Now with the stance's torque scripted: it starts on the stance's first tick.
    n_before = states.index(STANCE)
    leg, muscle = _leg([0.0] * n_before + STANCE_TORQUE)
    rows = _walk(leg, 2.7 + 0.15 + 1.0, strikes)
    states = [s for _, _, s, _ in rows]
    toe_off = n_before + len(STANCE_TORQUE) - 1
    assert states[n_before : toe_off + 1] == [STANCE] * len(STANCE_TORQUE), "stance through the toe-off tick"
    assert rows[toe_off][1] == pytest.approx(25.0 / (0.8 * 20.0) * 0.8 * 15.5), "the toe-off tick still commands"
    assert states[toe_off + 1] == REEL_OUT and rows[toe_off + 1][1] == 0.0, "reel-out on the next tick"
    reel_out = [k for k, s in enumerate(states) if s == REEL_OUT and k > toe_off]
    assert (reel_out[-1] - reel_out[0] + 1) / RATE == pytest.approx(0.2 + 1 / RATE, abs=1 / RATE)
    assert states[reel_out[-1] + 1] == SWING
    assert [d["in_stance"] for _, _, _, d in rows[n_before : toe_off + 2]] == [1.0] * len(STANCE_TORQUE) + [0.0]
    assert rows[toe_off][3]["stance_time"] == pytest.approx((len(STANCE_TORQUE) - 1) / RATE)
    assert rows[toe_off + 1][3]["stance_time"] == 0.0


def test_a_stance_under_5_n_m_lasts_through_swing_and_the_next_strike():
    """Nothing but the VNMC's toe-off ends its stance: not the next heel strike, not a lost gait phase."""
    leg, muscle = _leg([])
    strikes = [0.5, 1.6, 2.7]
    rows = _walk(leg, 2.7 + 0.2 + 0.1, strikes)
    n_before = [s for _, _, s, _ in rows].index(STANCE)
    leg, _ = _leg([0.0] * n_before + [3.0] * 2000)  # the raw torque never passes 5 N*m
    rows = _walk(leg, 8.0, strikes + [3.8])  # a fourth strike in stance, then none: the phase is lost after 2.4 s
    start = next(t for t, _, s, _ in rows if s == STANCE)
    tail = [(t, s, d) for t, _, s, d in rows if t > 2.7 + 0.2]
    assert all(s == STANCE for _, s, _ in tail)
    assert tail[-1][2]["phase_valid"] == 0.0, "the phase is gone, and stance goes on"
    assert tail[-1][2]["stance_time"] == pytest.approx(tail[-1][0] - start, abs=1e-9)


def test_reset_starts_the_leg_over():
    script = [0.0] * 420 + STANCE_TORQUE
    leg, muscle = _leg(list(script))
    first = _walk(leg, 4.0, [0.5, 1.6, 2.7])
    resets = muscle.resets
    leg.reset()
    muscle.script = list(script)
    again = _walk(leg, 4.0, [0.5, 1.6, 2.7])
    assert [r[:3] for r in first] == [r[:3] for r in again]
    assert muscle.resets == resets + 1, "the leg's reset resets its muscle"


def test_diagnostics_are_finite_and_carry_the_common_and_the_vnmc_keys():
    leg, _ = _leg([])
    common = {
        "torque_nm",
        "phase",
        "phase_valid",
        "in_stance",
        "heel_strike",
        "strike_time",
        "control_state",
        "stride_estimate",
    }
    before = leg.diagnostics()
    assert set(before) == common | set(VNMC_DIAGNOSTICS)
    assert before["control_state"] == REEL_OUT, "the state its first tick enters"
    for _, _, _, d in _walk(leg, 4.0, [0.5, 1.6, 2.7]):
        assert all(math.isfinite(v) for v in d.values())


def test_the_real_muscle_integrates_across_states_and_is_not_reset_at_stance_entry():
    """With the real muscle: the leg's muscle state equals one muscle stepped at 0.01 on the same angles up to the first
    stance tick, so nothing restarted it there."""
    leg = VNMCLeg(
        stance=VNMCStance(muscle=MusculoTendonJoint(timestep=1 / RATE), gain=1.468, peak_torque=25.0),
        heel_strike_detector=Strikes(),
        phase_estimator=StrideAverageGaitPhaseEstimator(),
        state_machine=BootStateMachine(reel_in_time=0.15, reel_out_time=0.2),
    )
    reference = MusculoTendonJoint(timestep=1 / RATE)
    ticks = np.arange(0.0, 3.2, 1 / RATE)
    hit = set(np.searchsorted(ticks, [0.5, 1.6, 2.7]).tolist())
    for k, t in enumerate(ticks):
        angle = -10.0 * math.sin(2 * math.pi * t / 1.1)
        leg.step(t, (k in hit, angle))
        if leg.state_machine.state == STANCE:
            reference.step(0.01 + 1.468 * reference.sensory_force, math.radians(angle))
            break
        reference.step(0.01, math.radians(angle))
    assert leg.state_machine.state == STANCE
    assert leg.stance.muscle.states() == reference.states()


# --- the device builder -----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    import mujoco

    from myoassist_utils.compose import compose_env_model

    return mujoco.MjModel.from_xml_string(compose_env_model("myolegs22", "DephyExoBoot_L1", terrain=None))


def _params(**overrides):
    from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig

    params = ExoImitationTrainSessionConfig.EnvParams.ExoControllerParams()
    for key, value in overrides.items():
        setattr(params, key, value)
    return params


def test_the_device_is_registered_and_built_at_its_rate(model):
    from myoassist_utils.exo_ctrl import DEVICE_BUILDERS, build_device_controller

    assert "exoboot_vnmc" in DEVICE_BUILDERS
    device = build_device_controller("exoboot_vnmc", _params(), model, physics_rate_hz=1200.0)
    assert [leg.side for leg in device.legs] == ["r", "l"]
    assert [model.actuator(i).name for i in device.actuator_ids] == ["Exo_R", "Exo_L"]
    assert device.schedule.rate_hz == 150.0
    for leg in device.legs:
        muscle = leg.controller.stance.muscle
        assert muscle.timestep == 1 / 150 and muscle.delay_steps == 3
        assert leg.controller.stance.gain == 1.468 and leg.controller.stance.peak_torque == 25.0
    shadow = build_device_controller("exoboot_vnmc", _params(shadow_mode=True), model, physics_rate_hz=1200.0)
    assert type(shadow).__name__ == "ShadowDevice" and len(shadow.legs) == 2


def test_each_leg_reads_its_foot_force_and_its_boots_ankle_angle(model):
    import mujoco

    from myoassist_utils.exo_ctrl.vnmc import LegSignals

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    sim = types.SimpleNamespace(data=data)
    for side, standing in (("r", -1.85), ("l", -8.48)):
        signals = LegSignals(model, side, standing_angle_deg=standing, rate_hz=150.0, quantize=False)
        force, angle = signals(sim)
        assert angle == pytest.approx(standing, abs=1e-9), "the standing keyframe reads the standing angle"
        assert force >= 0.0


def test_an_unknown_heel_strike_source_is_refused(model):
    from myoassist_utils.exo_ctrl.vnmc import build_vnmc_device

    with pytest.raises(ValueError, match="heel_strike_source"):
        build_vnmc_device(_params(heel_strike_source="gyro"), model, physics_rate_hz=1200.0)
