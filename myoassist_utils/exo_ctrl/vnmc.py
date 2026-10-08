"""The ExoBoot's virtual neuromuscular controller (VNMC): one Geyer-type muscle-tendon unit about the ankle.

Port of ``VirtualNeuroMuscularController`` (ExoBoot controllers.py:48-214), its muscle
(``muscle_model.MusculoTendonJoint``) and its branch of ``StanceSwingReeloutReelinStateMachine``, as the boot ran them in
the VNMC sessions. The committed code is an earlier snapshot; where its logs and it disagree, this follows the logs
(``tools/replay_vnmc_session.py`` checks each layer against them):

* **The muscle** is driven by the measured ankle angle (the raw encoder angle of that loop iteration; the logged
  ``filtered_ankle_angle`` is never written) and stimulated by positive force feedback, ``m_stim = 0.01 + VNMC_GAIN x
  F_mtu / F_max``, with the force from 20 ms earlier. It integrates on every loop iteration: in stance at that
  stimulation, in every other state at 0.01 (state_machines.py:236-238). It is never reset, so its state at a heel
  strike depends on the swing before it.
* **The torque filter, as it ran:** the boot passes the muscle's torque through an exponential moving average,
  ``trq_filt = FILT_ALPHA x trq + (1 - FILT_ALPHA) x trq_filt`` (FILT_ALPHA 0.8), but ``update_muscle_model`` sets
  ``trq_filt`` to 0 just before it, so it is ``0.8 x trq``, unsmoothed. Peak, toe-off and scaling all run on it.
* **The scaling, as it ran:** each stance's filtered torque is scaled so that the previous stance's peak would reach
  ``PEAK_TORQUE``: ``scalefactor = PEAK_TORQUE / previous stance's filtered peak`` (``PEAK_TORQUE / 100`` before the
  first toe-off) and the command is ``min(PEAK_TORQUE, scalefactor x filtered torque)``. The 0.8 cancels except before
  the first toe-off. The committed code has neither the filter nor PEAK_TORQUE (it hard-codes 25 N*m).
* **Toe-off from the muscle's own torque:** once the stance's filtered peak passes 5 N*m, eight ticks in a row with the
  torque not rising latch it, and it fires on the first tick after that on which the torque is at most 80% of the peak
  and not below the tick before. (The committed code waits four ticks; the logs show eight.) The state machine sees it
  on the next tick, as on the boot (``toe_off_switch_vnmc``), and reels out.

What runs around it is the boot's WALKING task: a heel strike (here from foot contact force, as for 4PTS) with a valid
stride-average gait phase starts reel-in; stance follows once reel-in ends; the VNMC's toe-off starts reel-out (0.2 s:
``SoftReelOutController`` with ``force_timer_to_complete``); swing follows. Only stance applies torque. Unlike the
boot's other stance controllers, losing the gait phase does not end a VNMC stance, and nothing but its toe-off does:
a stance whose filtered peak never passes 5 N*m (6.25 N*m of the muscle's) lasts through swing until a later rise of the torque ends it.

Plain Python floats throughout, since it runs inside the physics loop.
"""

from __future__ import annotations

import math

import mujoco

from myoassist_utils.exo_ctrl.base import HeelStrikeDetector
from myoassist_utils.exo_ctrl.boot_state import REEL_OUT, STANCE, AnkleEncoder, BootStateMachine
from myoassist_utils.exo_ctrl.factory import HEEL_STRIKE_SOURCES, LegExo
from myoassist_utils.exo_ctrl.phase import StrideAverageGaitPhaseEstimator, VgrfHeelStrikeDetector
from myoassist_utils.exo_ctrl.schedule import TickSchedule
from myoassist_utils.exo_ctrl.torque_adapter import ankle_torque_actuator

# The stimulation outside stance, and the floor of the reflex's.
IDLE_STIMULATION = 0.01
# Before the first toe-off the boot scales by PEAK_TORQUE over this ("Start w/ high number to ease user in").
INITIAL_REFERENCE_PEAK = 100.0
# The boot's FILT_ALPHA. Its moving average restarts from 0 every tick, so the filtered torque is FILT_ALPHA x torque.
FILT_ALPHA = 0.8
# VirtualNeuroMuscularController.toe_off_logic, on the filtered torque
TOE_OFF_MIN_PEAK = 5.0  # N*m, of the filtered torque (6.25 N*m of the muscle's)
TOE_OFF_FALLING_TICKS = 8
TOE_OFF_PEAK_FRACTION = 0.8
# SoftReelOutController(force_timer_to_complete=True), max_reel_out_time
VNMC_REEL_OUT_TIME = 0.2


class MusculoTendonJoint:
    """``muscle_model.MusculoTendonJoint`` with one joint, as ``VirtualNeuroMuscularController`` builds it.

    Its state is the contractile element's length ``l_ce`` and the activation ``A``; each ``step`` is one explicit Euler
    step of ``timestep`` (``MUSCLE_UPDATE_FREQUENCY`` 1, the only value the boot's code can run: with more substeps
    ``update`` calls ``update_inter`` with an argument it does not take). Within a step, in the boot's order:

    * activation: ``dA/dt = (S - A) / tau``, ``S`` clipped to [0.01, 1], tau 10 ms rising and 40 ms falling;
    * the MTU length from the joint angle, ``l_mtu = l_slack + l_opt - rho r (phi - phi_ref)``, so plantarflexion
      (phi > 0) shortens it; the series element is what the contractile element leaves of it;
    * force from the series element, ``F = F_max ((l_se / l_slack - 1) / e_ref)^2`` when stretched, from the length the
      contractile element had *before* this step;
    * the contractile velocity that balances that force against the force-length and force-velocity curves (Geyer
      2010), and ``l_ce`` advanced by it.

    The afferent delay is a queue of ``round(delay / timestep)`` samples of the normalized force: 4 at the boot's 175 Hz
    (20 ms rounds up to 22.9 ms), 3 at 150 Hz (exactly 20 ms). ``sensory_force`` is its oldest entry: read before a step,
    it is the force from that many steps earlier. It starts at rest at angle 0 (the boot's ``phi1_0``), activation 0.01.
    """

    # muscle_model.MusculoTendonJoint's class constants
    W = 0.56
    C = math.log(0.05)
    N = 1.5
    K = 5.0
    E_REF_PE = W
    E_REF_BE = 0.5 * W
    E_REF_BE2 = 1.0 - W
    TAU_ACT = 0.01
    TAU_DACT = 0.04
    S_MIN, S_MAX = 0.01, 1.0

    def __init__(
        self,
        *,
        timestep: float,
        f_max: float = 4000.0,
        l_opt: float = 0.04,
        v_max: float = 6.0,
        l_slack: float = 0.26,
        rho: float = 0.5,
        e_ref: float = 0.1,
        moment_arm: float = 0.04,
        phi_ref_deg: float = 0.0,
        initial_activation: float = 0.01,
        delay: float = 0.02,
    ):
        """The defaults are the boot's: ``VirtualNeuroMuscularController.__init__`` and its config's ``L_OPT``,
        ``V_MAX`` (in l_opt/s), ``L_SLACK``, ``E_REF`` and ``PHI_REF``. Lengths in m, force in N, the moment arm in m."""
        if not timestep > 0:
            raise ValueError(f"timestep must be > 0, got {timestep}")
        self.timestep = timestep
        self.f_max, self.l_opt, self.v_max, self.l_slack = f_max, l_opt, v_max, l_slack
        self.rho, self.e_ref, self.r = rho, e_ref, moment_arm
        self.phi_ref = phi_ref_deg * math.pi / 180
        self.initial_activation = initial_activation
        self.delay_steps = round(delay / timestep)
        if self.delay_steps < 1:
            raise ValueError(f"the afferent delay ({delay} s) is under half a step ({timestep} s)")
        self.reset()

    def reset(self) -> None:
        self.v_ce = 0.0
        self.f_mtu = 0.0
        self.activation = self.initial_activation
        self.l_ce = self.l_slack + self.l_opt - self.rho * self.r * (0.0 - self.phi_ref) - self.l_slack
        self._afferent = [0.0] * self.delay_steps  # normalized force, oldest first

    @property
    def sensory_force(self) -> float:
        """The normalized force ``delay_steps`` steps ago (``getSensoryData`` with ``F_mtu`` feedback)."""
        return self._afferent[0]

    @property
    def torque(self) -> float:
        """N*m, plantarflexion-positive: ``getTorque``."""
        return self.f_mtu * self.r

    def states(self) -> tuple[float, float, float]:
        """(force / F_max, l_ce / l_opt, v_ce / (l_opt v_max)): ``getStates``, what the boot logs as mtu_force, length_CE
        and velocity_CE."""
        return self.f_mtu / self.f_max, self.l_ce / self.l_opt, self.v_ce / (self.l_opt * self.v_max)

    def step(self, stimulation: float, angle_rad: float) -> None:
        """One step at ``stimulation`` with the joint at ``angle_rad`` (plantarflexion-positive)."""
        dt = self.timestep
        s = min(max(stimulation, self.S_MIN), self.S_MAX)
        a = self.activation
        tau = self.TAU_ACT if s > a else self.TAU_DACT
        a = a + (s - a) / tau * dt
        self.activation = a

        l_mtu = self.l_slack + self.l_opt
        l_mtu -= self.rho * self.r * (angle_rad - self.phi_ref)
        l_ce = self.l_ce
        f_se0 = _f_p0((l_mtu - l_ce) / self.l_slack, self.e_ref)
        f_be0 = _f_p0_ext(l_ce / self.l_opt, self.E_REF_BE, self.E_REF_BE2)
        f_pe0 = _f_p0(l_ce / self.l_opt, self.E_REF_PE)
        f_lce0 = math.exp(self.C * abs((l_ce / self.l_opt - 1) / self.W) ** 3)
        f_vce0 = (f_se0 + f_be0) / (f_pe0 + a * f_lce0)
        self.v_ce = self.l_opt * self.v_max * _inv_f_vce0(f_vce0, self.K, self.N)
        self.l_ce = l_ce + self.v_ce * dt
        self.f_mtu = self.f_max * f_se0

        self._afferent.append(self.f_mtu / self.f_max)
        self._afferent.pop(0)


def _inv_f_vce0(f_vce0: float, k: float, n: float) -> float:
    if f_vce0 <= 1:
        return (f_vce0 - 1) / (k * f_vce0 + 1)
    if f_vce0 <= n:
        temp = (f_vce0 - n) / (f_vce0 - n + 1)
        return (temp + 1) / (1 - 7.56 * k * temp)
    return 0.01 * (f_vce0 - n) + 1


def _f_p0(l0: float, e_ref: float) -> float:
    return ((l0 - 1) / e_ref) ** 2 if l0 > 1 else 0.0


def _f_p0_ext(l0: float, e_ref: float, e_ref2: float) -> float:
    return ((l0 - e_ref2) / e_ref) ** 2 if l0 < e_ref2 else 0.0


class TorqueToeOffDetector:
    """``VirtualNeuroMuscularController.toe_off_logic``: toe-off from the stance's own torque.

    Each tick, with the stance's peak so far (this tick's torque included): a tick whose torque is not above the last
    one counts as falling once the peak has passed ``TOE_OFF_MIN_PEAK``, and ``TOE_OFF_FALLING_TICKS`` in a row latch
    "descending". It fires on a tick that is descending, at most ``TOE_OFF_PEAK_FRACTION`` of the peak, and not below the
    last tick (the torque's first rise or plateau after its fall), and then unlatches. The last torque starts at -999,
    so the first tick of a stance is never falling.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._previous = -999.0
        self._falling = 0
        self._descending = False

    def step(self, torque: float, peak: float) -> bool:
        if torque <= self._previous and peak > TOE_OFF_MIN_PEAK:
            self._falling += 1
        else:
            self._falling = 0
        if self._falling >= TOE_OFF_FALLING_TICKS:
            self._descending = True
        fired = self._descending and torque <= TOE_OFF_PEAK_FRACTION * peak and self._previous <= torque
        if fired:
            self._descending = False
        self._previous = torque
        return fired


class VNMCStance:
    """``VirtualNeuroMuscularController`` as it ran: the muscle, its reflex, the per-stance scaling and the toe-off.

    Every tick steps the muscle exactly once: ``stance_tick`` in stance (``command``), ``idle_tick`` in any other state
    (the state machine's ``update_muscle_model(m_stim=0.01)``). ``start_stance`` is ``command(reset=True)``'s
    ``reset_constants``: it clears the toe-off detector and the stance's peak, not the muscle and not the scaling.
    ``stance_peak`` is the filtered torque's (``FILT_ALPHA``).
    """

    def __init__(self, *, muscle: MusculoTendonJoint, gain: float, peak_torque: float):
        if not (math.isfinite(gain) and gain >= 0):
            raise ValueError(f"the VNMC gain must be finite and >= 0, got {gain}")
        # The exo can only plantarflex, and the torque is clipped at peak_torque, so 0 would never assist.
        if not (math.isfinite(peak_torque) and peak_torque >= 0):
            raise ValueError(f"peak_torque must be finite and >= 0, got {peak_torque}")
        self.muscle = muscle
        self.gain = gain
        self.peak_torque = peak_torque
        self.toe_off_detector = TorqueToeOffDetector()
        self.reset()

    def reset(self) -> None:
        self.muscle.reset()
        self._reference_peak = INITIAL_REFERENCE_PEAK
        self.scalefactor = self.peak_torque / self._reference_peak
        self.stimulation = IDLE_STIMULATION
        self.command = 0.0
        self.toe_off = False
        self.start_stance()

    def start_stance(self) -> None:
        self.stance_peak = 0.0
        self.toe_off_detector.reset()

    def idle_tick(self, angle_deg: float) -> None:
        """Outside stance: the muscle at the idle stimulation, no torque."""
        self.stimulation = IDLE_STIMULATION
        self.muscle.step(IDLE_STIMULATION, math.radians(angle_deg))
        self.command = 0.0
        self.toe_off = False

    def stance_tick(self, angle_deg: float) -> float:
        """In stance: the reflex, one muscle step, the toe-off test and the scaled torque (N*m, plantarflexion-positive),
        in ``command``'s order. Sets ``toe_off`` on the tick it fires."""
        muscle = self.muscle
        self.stimulation = IDLE_STIMULATION + self.gain * muscle.sensory_force
        muscle.step(self.stimulation, math.radians(angle_deg))
        filtered = FILT_ALPHA * muscle.torque
        if filtered > self.stance_peak:
            self.stance_peak = filtered
        self.toe_off = self.toe_off_detector.step(filtered, self.stance_peak)
        if self.toe_off:
            self._reference_peak = self.stance_peak
        # Recomputed every stance tick, after the toe-off: the tick it fires already scales by this stance's peak.
        self.scalefactor = self.peak_torque / self._reference_peak
        self.command = min(self.peak_torque, self.scalefactor * filtered)
        return self.command


class VNMCLeg:
    """One leg: heel strikes -> stride-average gait phase -> the boot's state machine -> the VNMC, once per tick.

    In the boot's order: the gait-state estimator first (strike and phase), then the state machine, then the controller
    of the state it is in. The machine is ``BootStateMachine`` as it is, fed the VNMC's toe-off from the previous tick
    (the boot's ``toe_off_switch_vnmc`` is the stance controller's return value, read on the next step). For the VNMC the
    boot ends stance on that toe-off alone, so while in stance the machine is shown a gait phase even when there is none
    (``BootStateMachine`` would otherwise end stance on a lost phase, as the boot does for its other stance controllers).

    ``signal`` is the pair (heel-strike detector signal, ankle angle in the boot's deg, plantarflexion-positive).
    ``swing_only`` is the boot's SWING_ONLY, which holds the machine in swing; the env never sets it, a log replay does.
    """

    def __init__(
        self,
        *,
        stance: VNMCStance,
        heel_strike_detector: HeelStrikeDetector,
        phase_estimator: StrideAverageGaitPhaseEstimator,
        state_machine: BootStateMachine,
    ):
        self.stance = stance
        self.heel_strike_detector = heel_strike_detector
        self.phase_estimator = phase_estimator
        self.state_machine = state_machine
        self.swing_only = False
        self.reset()

    def reset(self) -> None:
        self.stance.reset()
        self.heel_strike_detector.reset()
        self.phase_estimator.reset()
        self.state_machine.reset()
        self._did_heel_strike = False
        self._phase: float | None = None
        self._toe_off_pending = False
        self._stance_start: float | None = None
        self._t = 0.0
        self._angle = 0.0

    def step(self, t: float, signal) -> float:
        contact, angle_deg = signal
        self._t, self._angle = t, angle_deg
        detector = self.heel_strike_detector
        self._did_heel_strike = detector.detect(t, contact)
        if self._did_heel_strike and detector.strike_time < t:
            # Dated back to the contact's onset (VgrfHeelStrikeDetector.min_contact_time), as for 4PTS.
            self.phase_estimator.estimate(detector.strike_time, True)
            self._phase = self.phase_estimator.estimate(t, False)
        else:
            self._phase = self.phase_estimator.estimate(t, self._did_heel_strike)
        machine = self.state_machine
        was = machine.state
        shown_phase = self._phase if (self._phase is not None or was != STANCE) else 0.0
        state = machine.step(
            t,
            did_heel_strike=self._did_heel_strike,
            did_toe_off=self._toe_off_pending,
            gait_phase=shown_phase,
            swing_only=self.swing_only,
            # Reel-in from when the foot landed, not from when the debounced detector confirmed it (as 4PTS dates its
            # stride): on the boot the gyro detector fires on the strike itself.
            strike_time=detector.strike_time if self._did_heel_strike else None,
        )
        if state == STANCE:
            if was != STANCE:
                self.stance.start_stance()
                self._stance_start = t
            torque = self.stance.stance_tick(angle_deg)
            self._toe_off_pending = self.stance.toe_off
        else:
            self.stance.idle_tick(angle_deg)
            self._stance_start = None
            self._toe_off_pending = False
            torque = 0.0
        return torque

    def diagnostics(self) -> dict[str, float]:
        # The keys every in-loop controller reports (as ExoBootFourPointSplineController.diagnostics), then the VNMC's
        # own: its muscle in the boot's logged units, its stimulation, the raw torque and the scaling, the time spent
        # in the current stance (0 outside one), and the ankle angle it read. Before the first tick the state is the
        # one that tick enters.
        stance, muscle = self.stance, self.stance.muscle
        state = self.state_machine.state
        mtu_force, length_ce, velocity_ce = muscle.states()
        # The first stride's duration is infinite (from no strike to the first), and averages so until it leaves the
        # window: no estimate yet either.
        stride = self.phase_estimator.mean_stride_duration
        return {
            "torque_nm": stance.command,
            "phase": -1.0 if self._phase is None else self._phase,
            "phase_valid": 0.0 if self._phase is None else 1.0,
            "in_stance": 1.0 if state == STANCE else 0.0,
            "heel_strike": 1.0 if self._did_heel_strike else 0.0,
            "strike_time": self.heel_strike_detector.strike_time if self._did_heel_strike else -1.0,
            "control_state": float(REEL_OUT if state is None else state),
            "stride_estimate": stride if stride is not None and math.isfinite(stride) else -1.0,
            "mtu_force": mtu_force,
            "length_ce": length_ce,
            "velocity_ce": velocity_ce,
            "vnmc_torque": muscle.torque,
            "m_stim": stance.stimulation,
            "scalefactor": stance.scalefactor,
            "stance_time": 0.0 if self._stance_start is None else self._t - self._stance_start,
            "ankle_angle_deg": self._angle,
        }


# The VNMC's own diagnostics() keys, beyond the ones every in-loop controller reports.
VNMC_DIAGNOSTICS = (
    "mtu_force",
    "length_ce",
    "velocity_ce",
    "vnmc_torque",
    "m_stim",
    "scalefactor",
    "stance_time",
    "ankle_angle_deg",
)


class LegSignals:
    """What one VNMC leg reads per tick: its foot force (for heel strikes) and its boot's ankle angle.

    The angle is a function of the joint position alone; the encoder's only state is its velocity filter, which the
    VNMC does not read, so nothing here needs resetting between episodes.
    """

    def __init__(self, model: mujoco.MjModel, side: str, *, standing_angle_deg: float, rate_hz: float, quantize: bool):
        from myoassist_utils.exo_ctrl.device import FootForce  # device.py registers this module's builder

        self.foot = FootForce(model, side)
        self.ankle = AnkleEncoder(model, side, standing_angle_deg=standing_angle_deg, sample_rate_hz=rate_hz, quantize=quantize)

    def __call__(self, sim) -> tuple[float, float]:
        return self.foot(sim), self.ankle.read(sim)[0]


def build_vnmc_leg(params, *, rate_hz: float) -> VNMCLeg:
    """One leg's VNMC from ``ExoControllerParams``-shaped ``params``, its muscle stepped at ``rate_hz``."""
    return VNMCLeg(
        stance=VNMCStance(
            muscle=MusculoTendonJoint(timestep=1.0 / rate_hz),
            gain=params.vnmc_gain,
            peak_torque=params.peak_torque,
        ),
        heel_strike_detector=VgrfHeelStrikeDetector(
            grf_on=params.grf_on_newtons,
            grf_off=params.grf_off_newtons,
            min_unload_time=params.min_unload_time,
            min_contact_time=params.min_contact_time,
        ),
        phase_estimator=StrideAverageGaitPhaseEstimator(
            num_strides_required=params.num_strides_required,
            num_strides_to_average=params.num_strides_to_average,
            min_stride_duration=params.min_stride_duration,
            max_stride_duration=params.max_stride_duration,
        ),
        state_machine=BootStateMachine(reel_in_time=params.reel_in_time, reel_out_time=params.reel_out_time),
    )


def build_vnmc_device(params, model: mujoco.MjModel, *, physics_rate_hz: float):
    """The ExoBoot VNMC at ``params.controller_rate_hz`` (150 Hz: every 8th substep of 1200 Hz physics), heel strikes
    from foot GRF, the muscle on the ankle angle as the boot's encoder reads it. The rate also sets the muscle's Euler
    step and its afferent delay in steps (``tools/replay_vnmc_session.py`` layer 6 measures what 150 Hz changes)."""
    from myoassist_utils.exo_ctrl.device import FixedRateLegExos  # device.py registers this builder

    if params.heel_strike_source not in HEEL_STRIKE_SOURCES:
        raise ValueError(f"unknown heel_strike_source {params.heel_strike_source!r}; expected one of {HEEL_STRIKE_SOURCES}")
    rate = params.controller_rate_hz
    legs, signals = [], []
    for side, actuator_name in (("r", params.right_actuator), ("l", params.left_actuator)):
        legs.append(
            LegExo(
                side=side,
                actuator_id=ankle_torque_actuator(model, actuator_name, side),
                controller=build_vnmc_leg(params, rate_hz=rate),
            )
        )
        signals.append(
            LegSignals(
                model,
                side,
                standing_angle_deg=getattr(params, f"ankle_standing_angle_{side}_deg"),
                rate_hz=rate,
                quantize=params.sensor_quantize,
            )
        )
    return FixedRateLegExos(legs, signals=signals, schedule=TickSchedule(rate_hz=rate, physics_rate_hz=physics_rate_hz))
