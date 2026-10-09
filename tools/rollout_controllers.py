"""Roll a trained policy out with an in-loop exo controller, and check it against what it should deliver.

The policy walks; the device env's controller drives the exo. Each controller's suite (``SUITES``) names its cases and
its checks. The cases run each in its own env and all from the same start points in the reference motion, so they
differ only in the exo. 4PTS's:

  exo off        device_controller "zero"
  4PTS sensing   4PTS with peak and bias torque 0: it detects strikes and estimates phase but applies nothing, so the
                 gait must be the exo-off gait exactly
  4PTS           the config's 4PTS

Every physics substep is recorded: each leg's commanded torque, the torque MuJoCo applied, foot force, and, from the
controller's ``diagnostics()``, its strike, phase, phase-valid and stance flags and its control state. The same 4PTS is
also run on every substep of that record (``physics_rate_pass``): the controller at the physics rate, the reference the
150 Hz one is held to. Checked:

  sensing   every stance (a contact lasting ``MIN_STANCE`` or more) is a strike in the loop, dated within a tick of its
            onset, and nothing else is -- such as a foot scuffing the ground in swing; so no stride is split; stride
            time and stance fraction; phase valid from the third strike
  torque    applied = commanded; torque changes only on ticks; per stride, the delivered torque lags the physics-rate
            controller's by under a tick, with peak and impulse within a few percent
  gait      episode length, ending, speed and ankle angle against exo off. Reported, not required: the policy was not
            trained with this assistance.

Episodes run in evaluate mode at the config's target speed, and start at chosen indices of the reference motion
(``reset_at``): the env's own reset draws the index from an unseeded generator. Physics and the deterministic policy
are otherwise fixed, so each episode is reproducible from its index. Every ``--index-step``-th start index is tried
with the exo off, and the longest that walk for at least ``--min-seconds`` (up to ``--max-kept``) are run in every
case.

Run from the repo root (configs name the reference data relative to it):

    python tools/rollout_controllers.py <policy.zip> [--controller 4PTS] [--config <its config>] [--min-seconds 8]
        [--out <dir>]

Writes report.md, torque_vs_phase.png and episodes.npz to --out (default rl_train/results/rollouts/<time>).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import pathlib
from collections.abc import Callable, Sequence

import numpy as np

from myoassist_utils.exo_ctrl import TickSchedule
from myoassist_utils.exo_ctrl.device import FootForce
from myoassist_utils.exo_ctrl.factory import build_leg_exos
from myoassist_utils.exo_ctrl.fourpoint_spline import FourPointSpline
from myoassist_utils.exo_ctrl.torque_adapter import PLANTARFLEXION_SIGN

DEFAULT_CONFIG = pathlib.Path("rl_train/train/train_configs/exoboot_spline/imitation_22_DephyExoBoot_L1_exoboot_spline.json")
DEVICE_ENV_ID = "myoAssistLegImitationExoDevice-v0"
SIDES = ("r", "l")  # the device controllers' actuator order: right, then left
# The case every controller is compared against, and whose sweep picks the start indices.
EXO_OFF = "exo off"
EXO_OFF_CASE = dict(device_controller="zero")
STEP_JOINTS = ("pelvis_tx", "pelvis_ty", "ankle_angle_r", "ankle_angle_l")
# Plot colors: the first two slots of the dataviz skill's validated categorical palette, then its ink and muted grays.
DELIVERED, REFERENCE, INK, MUTED = "#2a78d6", "#eb6834", "#52514e", "#898781"


# --- recording -------------------------------------------------------------------------------------------------------


class Recorder:
    """Wraps the env's device controller and records every physics substep, before the sim advances by it.

    Torques are recorded plantarflexion-positive, as the controllers think of them. ``applied_*`` is the joint torque
    MuJoCo applied over that substep (``actuator_force * gear``); it is read on the next call, once the sim has
    computed it, and re-aligned by ``take``.
    """

    # Each leg's fields after the torques and foot forces, and the ``diagnostics()`` key each is read from. A key the
    # controller does not report is NaN, except the strike time, which is then the time of the row reporting the strike.
    # Diagnostics are read on every substep: between ticks they are the last tick's.
    DIAGNOSTICS = (
        ("strike", "heel_strike"),
        ("phase", "phase"),
        ("valid", "phase_valid"),
        ("stance", "in_stance"),
        ("strike_time", "strike_time"),
        ("control_state", "control_state"),
        ("stride_estimate", "stride_estimate"),
    )
    FIELDS = (
        "t",
        "tick",
        *(f"{name}_{side}" for name in ("cmd", "applied", "grf") for side in SIDES),
        *(f"{name}_{side}" for name, _ in DIAGNOSTICS for side in SIDES),
    )

    def __init__(self, inner, model, extra_keys: Sequence[str] = ()):
        """``extra_keys``: more ``diagnostics()`` keys to record, as ``<key>_<side>`` fields after ``FIELDS``."""
        self.inner = inner
        self.extra_keys = tuple(extra_keys)
        self.fields = self.FIELDS + tuple(f"{key}_{side}" for key in self.extra_keys for side in SIDES)
        self.actuator_ids = tuple(inner.actuator_ids)
        self._gear = [float(model.actuator_gear[a, 0]) for a in self.actuator_ids]
        self._feet = [FootForce(model, side) for side in SIDES]
        self._legs = list(getattr(inner, "legs", []))
        # The same schedule as the controller's, to know which substeps are ticks without touching its own.
        schedule = getattr(inner, "schedule", None)
        self._schedule = (
            None if schedule is None else TickSchedule(rate_hz=schedule.rate_hz, physics_rate_hz=schedule.physics_rate_hz)
        )
        self._rows: list[list[float]] = []

    def reset(self) -> None:
        self.inner.reset()
        if self._schedule is not None:
            self._schedule.reset()
        self._rows = []

    def compute_torque(self, sim):
        data = sim.data
        applied_before = [
            PLANTARFLEXION_SIGN * float(data.actuator_force[a]) * g for a, g in zip(self.actuator_ids, self._gear)
        ]
        torques = self.inner.compute_torque(sim)
        tick = self._schedule.tick() if self._schedule is not None else True
        row = [float(data.time), float(tick), *(PLANTARFLEXION_SIGN * float(x) for x in torques), *applied_before]
        row += [foot(sim) for foot in self._feet]
        if self._legs:
            diagnostics = [leg.controller.diagnostics() for leg in self._legs]
            for _, key in self.DIAGNOSTICS:
                if key == "strike_time":
                    # When the strike reported on this tick happened: a debounced detector dates it back to the
                    # contact's onset.
                    row += [d.get(key, row[0]) if d.get("heel_strike") == 1 else np.nan for d in diagnostics]
                else:
                    row += [d.get(key, np.nan) for d in diagnostics]
            row += [d.get(key, np.nan) for key in self.extra_keys for d in diagnostics]
        else:
            row += [np.nan] * (len(self.fields) - len(row))
        self._rows.append(row)
        return torques

    def take(self) -> dict[str, np.ndarray]:
        """The episode's record, one array per field, and a fresh start."""
        table = np.array(self._rows, dtype=float).reshape(-1, len(self.fields))
        record = {name: table[:, i] for i, name in enumerate(self.fields)}
        for side in SIDES:
            applied = np.full(len(table), np.nan)
            applied[:-1] = record[f"applied_{side}"][1:]  # read one substep later; the last one is never read
            record[f"applied_{side}"] = applied
        self._rows = []
        return record


@dataclasses.dataclass
class Episode:
    case: str
    start_index: int
    ending: str
    dt: float
    steps: dict[str, np.ndarray]  # per control step, after it
    substeps: dict[str, np.ndarray]  # per physics substep (Recorder.FIELDS)

    @property
    def duration(self) -> float:
        return len(self.steps["t"]) * self.dt


def reset_at(env, index: int):
    """``MyoAssistLegImitation.reset``, from reference index ``index`` instead of one drawn at random."""
    env._imitation_index = int(index)
    env._imitation_index_exact = float(index)
    env._start_target_velocity_schedule()  # first: the start state scales the reference velocities by it
    env._set_episode_start_state()
    return env._reset_simulation(reset_qpos=env.sim.data.qpos, reset_qvel=env.sim.data.qvel)


def run_episode(env, policy, recorder: Recorder, index: int, *, case: str, max_steps: int, safe_height: float) -> Episode:
    obs, _ = reset_at(env, index)
    steps = {key: [] for key in ("t", *STEP_JOINTS)}
    ending = "time limit"
    for _ in range(max_steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        data = env.sim.data
        steps["t"].append(float(data.time))
        for joint in STEP_JOINTS:
            steps[joint].append(float(data.joint(joint).qpos[0]))
        if not np.all(np.isfinite(obs)):
            ending = "non-finite observation"
            break
        if terminated:
            ending = "fell" if steps["pelvis_ty"][-1] < safe_height else "off the reference"
            break
        if truncated:
            break
    return Episode(case, int(index), ending, float(env.dt), {k: np.array(v) for k, v in steps.items()}, recorder.take())


def make_env(config_path: pathlib.Path, case: dict, extra_keys: Sequence[str] = ()):
    """The device env for one case, with a ``Recorder`` around its controller (recording ``extra_keys`` too). Returns
    (env, config)."""
    from rl_train.envs.environment_handler import EnvironmentHandler

    env_id = json.loads(pathlib.Path(config_path).read_text())["env_params"]["env_id"]
    if env_id != DEVICE_ENV_ID:
        raise ValueError(f"{config_path} uses env_id {env_id!r}; the in-loop controllers need {DEVICE_ENV_ID!r}")
    config = EnvironmentHandler.get_session_config_from_path(
        str(config_path), EnvironmentHandler.get_config_type_from_session_id(env_id)
    )
    config.env_params.num_envs = 1
    overrides = dict(case)
    config.env_params.device_controller = overrides.pop("device_controller")
    for key, value in overrides.items():
        setattr(config.env_params.exo_controller_params, key, value)
    # Evaluate mode, at one fixed target speed, as gait_evaluate.py runs it, so every episode is asked the same speed.
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False, is_evaluate_mode=True)
    speed = (config.env_params.min_target_velocity + config.env_params.max_target_velocity) / 2
    env.set_target_velocity_mode_manually(type(env).VelocityMode.UNIFORM, 0.0, speed, speed, speed)
    model = getattr(env.sim.model, "ptr", env.sim.model)
    env.set_device_controller(Recorder(env.device_controller, model, extra_keys))
    return env, config


def load_policy(path):
    import stable_baselines3
    import torch

    from rl_train.train.policies.rl_agent_exo import HumanExoActorCriticPolicy

    torch.set_num_threads(1)  # one small MLP per step: more threads only add overhead
    return stable_baselines3.PPO.load(str(path), custom_objects={"policy_class": HumanExoActorCriticPolicy}, device="cpu")


# --- analysis ----------------------------------------------------------------------------------------------------------


# A contact at least this long is a stance; anything shorter is a touch, such as a foot scuffing the ground in swing.
# On the MyoAssist tutorial policy scuffs last 2-30 ms and stances at least 150 ms.
MIN_STANCE = 0.1


def physics_rate_pass(substeps: dict[str, np.ndarray], params, model) -> dict[str, np.ndarray]:
    """4PTS run on every substep from the recorded foot force -- the controller at the physics rate. Its torque per
    side, on each substep."""
    t = substeps["t"]
    out = {}
    for leg in build_leg_exos(params, model):
        leg.controller.reset()
        out[leg.side] = np.array([leg.controller.step(float(a), float(f)) for a, f in zip(t, substeps[f"grf_{leg.side}"])])
    return out


def contacts(t: np.ndarray, grf: np.ndarray, *, on: float, off: float, min_unload: float) -> tuple[np.ndarray, np.ndarray]:
    """Every foot contact, by the heel-strike detector's hysteresis and unload rule applied to each sample, but with no
    minimum duration. Returns each contact's onset and end (when the unload that released it began; NaN if the contact
    was still on at the last sample). A foot loaded at the first sample has no onset, as for the detector.
    """
    onsets, ends = [], []
    in_contact, onset, unload = grf[0] > off, None, None
    for ti, fi in zip(t, grf):
        if not in_contact:
            if fi > on:
                in_contact, onset, unload = True, ti, None
            continue
        if fi >= off:
            unload = None
        elif unload is None:
            unload = ti
        if unload is not None and ti - unload >= min_unload:
            if onset is not None:
                onsets.append(onset)
                ends.append(unload)
            in_contact, onset = False, None
    if in_contact and onset is not None:
        onsets.append(onset)
        ends.append(np.nan)
    return np.array(onsets), np.array(ends)


def split_contacts(onsets: np.ndarray, ends: np.ndarray, t_end: float, min_stance: float = MIN_STANCE):
    """(stance onsets, stance ends, touch onsets, undetermined onsets).

    A contact still on at ``t_end`` is a stance if it has already lasted long enough; if not, it is undetermined -- the
    record ended before it could be told apart from a touch.
    """
    still_on = np.isnan(ends)
    durations = np.where(still_on, t_end - onsets, ends - onsets)
    stance = durations >= min_stance
    touch = ~stance & ~still_on
    return onsets[stance], ends[stance], onsets[touch], onsets[~stance & still_on]


def strike_delays(found: np.ndarray, expected: np.ndarray, *, max_delay: float) -> tuple[np.ndarray, int, np.ndarray]:
    """Match each expected strike with the first found strike at or after it, at most ``max_delay`` later.

    Returns the delays of the matched ones, the number of expected strikes left unmatched (missed), and the found
    strikes matching none (false strikes).
    """
    found = np.sort(found)
    used = np.zeros(len(found), dtype=bool)
    delays, missed = [], 0
    for strike in expected:
        j = int(np.searchsorted(found, strike - 1e-9))
        if j < len(found) and not used[j] and found[j] - strike <= max_delay:
            used[j] = True
            delays.append(found[j] - strike)
        else:
            missed += 1
    return np.array(delays), missed, found[~used]


def stride_table(strikes: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per stride between consecutive strikes: its duration, and its stance fraction (contact end over duration)."""
    durations = np.diff(strikes)
    fractions = []
    for start, stop in zip(strikes[:-1], strikes[1:]):
        end = ends[(ends > start) & (ends <= stop)]
        fractions.append((end[0] - start) / (stop - start) if len(end) else np.nan)
    return durations, np.array(fractions)


def stride_average(t: np.ndarray, values: np.ndarray, strikes: np.ndarray, bins: int = 100):
    """``values`` over each complete stride, averaged into ``bins`` bins of that stride's own phase (0 at its strike,
    1 at the next). Returns (bin centres, an array of one row per stride)."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for start, stop in zip(strikes[:-1], strikes[1:]):
        inside = (t >= start) & (t < stop)
        index = np.minimum(((t[inside] - start) / (stop - start) * bins).astype(int), bins - 1)
        sums = np.bincount(index, weights=values[inside], minlength=bins)
        counts = np.bincount(index, minlength=bins)
        rows.append(np.where(counts > 0, sums / np.maximum(counts, 1), np.nan))
    return (edges[:-1] + edges[1:]) / 2, np.array(rows).reshape(-1, bins)


def compare_strides(t: np.ndarray, delivered: np.ndarray, reference: np.ndarray, strikes: np.ndarray):
    """Per complete stride: the delivered torque's lag behind the reference (difference of torque-weighted mean times,
    ms), and its peak and impulse as fractions above the reference's. NaN for a stride in which the reference applies no
    torque, so entry k is always stride k.

    Samples are evenly spaced, so sums stand in for integrals.
    """
    lags, peaks, impulses = [], [], []
    for start, stop in zip(strikes[:-1], strikes[1:]):
        inside = (t >= start) & (t < stop)
        d, r, ts = delivered[inside], reference[inside], t[inside]
        if r.sum() <= 0:
            lags.append(np.nan)
            peaks.append(np.nan)
            impulses.append(np.nan)
            continue
        lags.append((np.sum(d * ts) / d.sum() - np.sum(r * ts) / r.sum()) * 1e3 if d.sum() > 0 else np.nan)
        peaks.append(d.max() / r.max() - 1)
        impulses.append(d.sum() / r.sum() - 1)
    return np.array(lags), np.array(peaks), np.array(impulses)


def dated_within(found: np.ndarray, expected: np.ndarray, tolerance: float) -> np.ndarray:
    """For each expected strike, whether a found strike is dated at or after it, at most ``tolerance`` later."""
    if not len(found):
        return np.zeros(len(expected), dtype=bool)
    found = np.sort(found)
    j = np.searchsorted(found, expected - 1e-9)
    later = np.where(j < len(found), found[np.minimum(j, len(found) - 1)] - expected, np.inf)
    return later <= tolerance


def changes_off_tick(values: np.ndarray, tick: np.ndarray) -> int:
    """Substeps whose value differs from the one before although the controller did not tick there."""
    return int(np.count_nonzero((np.diff(values) != 0) & (tick[1:] == 0)))


def gated_profile(phase: np.ndarray, params, stride: float) -> np.ndarray:
    """What 4PTS delivers at ``phase`` of a steady ``stride``-long gait: the spline from reel-in to toe-off, else 0."""
    spline = FourPointSpline(
        rise_fraction=params.rise_fraction,
        peak_fraction=params.peak_fraction,
        fall_fraction=params.fall_fraction,
        peak_torque=params.peak_torque,
        bias_torque=params.bias_torque,
        peak_hold_time=params.peak_hold_time,
    )
    on = (phase >= params.reel_in_time / stride) & (phase <= params.toe_off_fraction)
    return np.where(on, [spline.torque(p) for p in phase], 0.0)


def _fmt(values, scale=1.0, digits=1, unit=""):
    values = np.asarray(values, dtype=float) * scale
    values = values[np.isfinite(values)]
    if not len(values):
        return "n/a"
    return f"{values.mean():+.{digits}f} ± {values.std():.{digits}f}{unit}"


def analyze(results: dict[str, list[Episode]], params, model, rate_hz: float) -> dict:
    """Every check, per leg, over all kept episodes."""
    tick = 1.0 / rate_hz
    a = {"tick_s": tick, "min_stride": params.min_stride_duration, "legs": {}, "gait": [], "nonfinite": []}

    def stances(ep, side):
        s = ep.substeps
        onsets, ends = contacts(
            s["t"], s[f"grf_{side}"], on=params.grf_on_newtons, off=params.grf_off_newtons, min_unload=params.min_unload_time
        )
        return split_contacts(onsets, ends, s["t"][-1])

    for side in SIDES:
        leg = dict(delays=[], missed=0, false=0, false_on_touch=0, touches=0, loop_strides=[], strides=[], fractions=[])
        leg.update(valid_after_third=[], valid_share=[], applied_error=0.0, off_tick=0, lags=[], peaks=[], impulses=[])
        leg.update(profiles=[], peak_phase=[], late_dated=0)
        for ep in results["4PTS sensing"]:
            s = ep.substeps
            stance_onsets, stance_ends, touches, undetermined = stances(ep, side)
            reported = (s["tick"] == 1) & (s[f"strike_{side}"] == 1)
            loop = s[f"strike_time_{side}"][reported]  # when each strike the controller reported happened
            # A strike on a contact the record cut short cannot be judged either way.
            loop = np.array([f for f in loop if not np.any(np.abs(undetermined - f) <= tick + 1e-9)])
            delays, missed, false = strike_delays(loop, stance_onsets, max_delay=tick + 1e-9)
            leg["delays"] += list(delays)
            leg["missed"] += missed
            leg["false"] += len(false)
            leg["false_on_touch"] += sum(bool(np.any(np.abs(touches - f) <= tick + 1e-9)) for f in false)
            leg["touches"] += len(touches)
            leg["loop_strides"] += list(np.diff(loop))
            durations, fractions = stride_table(stance_onsets, stance_ends)
            leg["strides"] += list(durations)
            leg["fractions"] += list(fractions)
            valid = s[f"valid_{side}"] == 1
            reported_at = s["t"][reported]
            if len(reported_at) >= 3 and valid.any():
                first_valid = s["t"][np.argmax(valid)]
                leg["valid_after_third"].append(first_valid - reported_at[2])
                leg["valid_share"].append(valid[s["t"] >= first_valid].mean())
        for ep in results["4PTS"]:
            s = ep.substeps
            stance_onsets = stances(ep, side)[0]
            reference = physics_rate_pass(s, params, model)[side]
            cmd, applied = s[f"cmd_{side}"], s[f"applied_{side}"]
            leg["applied_error"] = max(leg["applied_error"], float(np.nanmax(np.abs(applied - cmd))))
            leg["off_tick"] += changes_off_tick(cmd, s["tick"])
            lags, peaks, impulses = compare_strides(s["t"], cmd, reference, stance_onsets)
            # Timing is compared where both controllers start the stride at its onset. Where the loop dates the
            # strike later -- a first touch too brief for a tick to see, the foot then resting lightly until it
            # lands -- the two run different strides, which is a sampling difference, not the tick's delay.
            loop = s[f"strike_time_{side}"][(s["tick"] == 1) & (s[f"strike_{side}"] == 1)]
            same_start = dated_within(loop, stance_onsets[:-1], tick + 1e-9)
            assisted = np.isfinite(lags)
            leg["late_dated"] += int(np.count_nonzero(assisted & ~same_start))
            leg["lags"] += list(lags[assisted & same_start])
            leg["peaks"] += list(peaks[assisted & same_start])
            leg["impulses"] += list(impulses[assisted & same_start])
            phase, rows = stride_average(s["t"], cmd, stance_onsets)
            assisted = np.nanmax(rows, axis=1) > 0 if len(rows) else np.zeros(0, dtype=bool)
            leg["profiles"] += list(rows[assisted])
            leg["peak_phase"] += list(phase[np.nanargmax(rows[assisted], axis=1)]) if assisted.any() else []
            leg["phase_bins"] = phase
        a["legs"][side] = leg
    for case, eps in results.items():
        for ep in eps:
            # The controller flags are placeholders (NaN) with the exo off, and the last applied torque is never read.
            recorded = [ep.substeps[k] for k in ("t", "cmd_r", "cmd_l", "grf_r", "grf_l")] + list(ep.steps.values())
            finite = all(np.all(np.isfinite(v)) for v in recorded)
            if not finite or ep.ending == "non-finite observation":
                a["nonfinite"].append((case, ep.start_index))
            strides = np.concatenate([np.diff(stances(ep, side)[0]) for side in SIDES])
            speed = (ep.steps["pelvis_tx"][-1] - ep.steps["pelvis_tx"][0]) / max(ep.duration - ep.dt, ep.dt)
            a["gait"].append(
                dict(
                    case=case,
                    index=ep.start_index,
                    duration=ep.duration,
                    ending=ep.ending,
                    speed=speed,
                    stride=float(np.mean(strides)) if len(strides) else np.nan,
                    plantarflexion=-np.degrees(min(ep.steps["ankle_angle_r"].min(), ep.steps["ankle_angle_l"].min())),
                )
            )
    a["same_as_exo_off"] = max(
        (
            float(np.max(np.abs(off.steps[j] - on.steps[j]))) if len(off.steps[j]) == len(on.steps[j]) else np.inf
            for off, on in zip(results[EXO_OFF], results["4PTS sensing"])
            for j in STEP_JOINTS
        ),
        default=np.nan,
    )
    return a


# --- outputs -----------------------------------------------------------------------------------------------------------


def plot(results: dict[str, list[Episode]], analysis: dict, params, model, path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 8.5), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[1.3, 1.0, 0.7])
    strides = [d for side in SIDES for d in analysis["legs"][side]["strides"]]
    mean_stride = float(np.mean(strides)) if strides else 1.1
    fine = np.linspace(0, 1, 1001)
    for col, side in enumerate(SIDES):
        ax = fig.add_subplot(grid[0, col])
        leg = analysis["legs"][side]
        if leg["profiles"]:
            rows = np.array(leg["profiles"])
            phase, mean, sd = leg["phase_bins"], np.nanmean(rows, axis=0), np.nanstd(rows, axis=0)
            ax.fill_between(phase, mean - sd, mean + sd, color=DELIVERED, alpha=0.18, linewidth=0)
            ax.plot(phase, mean, color=DELIVERED, linewidth=2, label=f"delivered, mean ± sd of {len(rows)} strides")
        ax.plot(fine, gated_profile(fine, params, mean_stride), color=REFERENCE, linewidth=2, label="4PTS, steady gait")
        ax.set(title=f"{'Right' if side == 'r' else 'Left'} ankle", xlabel="gait phase (heel strike to heel strike)")
        ax.set_ylabel("plantarflexion torque (N·m)")
        ax.set_xlim(0, 1)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    eps = results["4PTS"]
    if eps:
        ep = max(eps, key=lambda e: e.duration)
        s = ep.substeps
        reference = physics_rate_pass(s, params, model)["r"]
        onsets, _ = contacts(
            s["t"], s["grf_r"], on=params.grf_on_newtons, off=params.grf_off_newtons, min_unload=params.min_unload_time
        )
        later = onsets[onsets > s["t"][0] + 4.0]
        t0 = later[0] if len(later) else s["t"][0]
        window = (s["t"] >= t0) & (s["t"] <= t0 + 3.0)
        ax = fig.add_subplot(grid[1, :])
        ax.plot(s["t"][window], reference[window], color=REFERENCE, linewidth=2, label="controller at 1200 Hz")
        ax.step(s["t"][window], s["cmd_r"][window], where="post", color=DELIVERED, linewidth=2, label="in the loop at 150 Hz")
        ax.set(title=f"Right ankle, 3 s of episode {ep.start_index}", ylabel="torque (N·m)")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        ax2 = fig.add_subplot(grid[2, :], sharex=ax)
        ax2.plot(s["t"][window], s["grf_r"][window], color=INK, linewidth=1.5)
        for level in (params.grf_on_newtons, params.grf_off_newtons):
            ax2.axhline(level, color=MUTED, linewidth=1, linestyle="--")
        ax2.set(ylabel="foot force (N)", xlabel="time (s)")
    for ax in fig.axes:
        ax.grid(color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _pass(ok: bool) -> str:
    return "pass" if ok else "**FAIL**"


def report(analysis: dict, *, policy, config_path, sweep: list[tuple[int, float]], kept: list[int], rate_hz: float) -> str:
    tick_ms = 1e3 / rate_hz
    legs = analysis["legs"]
    lines = [
        "# 4PTS rollouts",
        "",
        f"Policy `{policy}`, config `{config_path}`, {datetime.date.today().isoformat()}.",
        "",
        f"{len(sweep)} start indices tried with the exo off; {len(kept)} walked long enough and were run in every case: "
        + ", ".join(f"{i} ({d:.1f} s)" for i, d in sweep if i in kept)
        + ".",
        "",
        "## Pass bar",
        "",
        "| check | right | left | |",
        "|---|---|---|---|",
    ]

    def row(name, fmt, ok):
        cells = [fmt(legs[side]) for side in SIDES]
        lines.append(f"| {name} | {cells[0]} | {cells[1]} | {_pass(all(ok(legs[side]) for side in SIDES))} |")

    lines.append(f"| no NaN or crash | | | {_pass(not analysis['nonfinite'])} |")
    row(
        f"sensing: every stance (a contact of {MIN_STANCE * 1e3:.0f} ms or more) a strike, dated within a tick",
        lambda leg: (
            f"{len(leg['delays'])} of {len(leg['delays']) + leg['missed']}; delay {_fmt(leg['delays'], 1e3, unit=' ms')}"
        ),
        lambda leg: leg["missed"] == 0 and len(leg["delays"]) > 0,
    )
    row(
        "sensing: no other strike, such as a scuff in swing",
        lambda leg: f"{leg['false']} ({leg['false_on_touch']} on the {leg['touches']} shorter touches)",
        lambda leg: leg["false"] == 0,
    )
    min_stride = analysis["min_stride"]
    row(
        f"sensing: no stride under {min_stride:g} s between the controller's strikes",
        lambda leg: f"shortest {min(leg['loop_strides'], default=np.nan):.2f} s of {len(leg['loop_strides'])}",
        lambda leg: bool(leg["loop_strides"]) and min(leg["loop_strides"]) > min_stride,
    )
    gait_same = analysis["same_as_exo_off"]
    lines.append(f"| sensing gait identical to exo off | max difference {gait_same:.1e} | | {_pass(gait_same == 0)} |")
    row(
        "applied torque = command",
        lambda leg: f"max error {leg['applied_error']:.1e} N·m",
        lambda leg: leg["applied_error"] < 1e-6,
    )
    row(
        "torque held between ticks",
        lambda leg: f"{leg['off_tick']} changes off a tick",
        lambda leg: leg["off_tick"] == 0,
    )
    row(
        f"timing vs the controller at 1200 Hz (within {tick_ms:.1f} ms)",
        lambda leg: f"lag {_fmt(leg['lags'], unit=' ms')} over {len(leg['lags'])} strides",
        lambda leg: bool(leg["lags"]) and abs(np.nanmean(leg["lags"])) <= tick_ms,
    )
    lines += [
        "",
        "## Sensing (4PTS with no torque)",
        "",
        "| | right | left |",
        "|---|---|---|",
        "| stride (s) | " + " | ".join(_fmt(legs[s]["strides"], digits=3) for s in SIDES) + " |",
        "| stance fraction | " + " | ".join(_fmt(legs[s]["fractions"], digits=3) for s in SIDES) + " |",
        "| first valid phase, after the third strike | "
        + " | ".join(_fmt(legs[s]["valid_after_third"], 1e3, unit=" ms") for s in SIDES)
        + " |",
        "| share of ticks with a valid phase after that | "
        + " | ".join(_fmt(legs[s]["valid_share"], 100, unit="%") for s in SIDES)
        + " |",
        "",
        "## Torque (4PTS)",
        "",
        "Per stride, against the same controller run on every 1200 Hz substep of the same foot force:",
        "",
        "| | right | left |",
        "|---|---|---|",
        "| lag (ms) | " + " | ".join(_fmt(legs[s]["lags"]) for s in SIDES) + " |",
        "| peak | " + " | ".join(_fmt(legs[s]["peaks"], 100, unit="%") for s in SIDES) + " |",
        "| impulse | " + " | ".join(_fmt(legs[s]["impulses"], 100, unit="%") for s in SIDES) + " |",
        "| phase of the peak, on the true stride | " + " | ".join(_fmt(legs[s]["peak_phase"], digits=3) for s in SIDES) + " |",
        "| strides left out: the loop dated the strike more than a tick after the onset | "
        + " | ".join(str(legs[s]["late_dated"]) for s in SIDES)
        + " |",
        "",
        "## Gait",
        "",
        "Reported, not required: the policy was not trained with this assistance.",
        "",
        "| case | start index | duration (s) | ending | speed (m/s) | stride (s) | peak plantarflexion (°) |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in analysis["gait"]:
        lines.append(
            f"| {g['case']} | {g['index']} | {g['duration']:.1f} | {g['ending']} | {g['speed']:.2f} | {g['stride']:.2f} "
            f"| {g['plantarflexion']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def save_episodes(results: dict[str, list[Episode]], path: pathlib.Path) -> None:
    arrays = {}
    for case, eps in results.items():
        for ep in eps:
            prefix = f"{case.replace(' ', '_')}/{ep.start_index}"
            arrays.update({f"{prefix}/step/{k}": v for k, v in ep.steps.items()})
            arrays.update({f"{prefix}/substep/{k}": v for k, v in ep.substeps.items()})
    np.savez_compressed(path, **arrays)


# --- VNMC ------------------------------------------------------------------------------------------------------------

VNMC_CONFIG = pathlib.Path("rl_train/train/train_configs/exoboot_vnmc/imitation_22_DephyExoBoot_L1_exoboot_vnmc.json")
VNMC_SHADOW, VNMC_ASSIST = "VNMC shadow", "VNMC"
# The would-be command (recorded even in shadow mode, where the applied one is zero) and the VNMC's own diagnostics.
VNMC_EXTRA_KEYS = (
    "torque_nm",
    "mtu_force",
    "length_ce",
    "velocity_ce",
    "vnmc_torque",
    "m_stim",
    "scalefactor",
    "stance_time",
    "ankle_angle_deg",
)
# The muscle-side quantities whose ranges are compared with the boot's logs (VNMC_LOG_RANGES).
VNMC_RANGE_KEYS = ("ankle_angle_deg", "mtu_force", "length_ce", "velocity_ce", "vnmc_torque", "m_stim", "scalefactor")
# On the VNMC sessions' logs, per leg (p1 and p99 of every row while walking at constant parameters, the mean of the two
# sessions), for the report to set the policy's ranges against: the muscle follows the absolute ankle angle, so a gait
# with another ankle range, or another standing angle, drives it elsewhere.
VNMC_LOG_RANGES = {
    "r": {
        "ankle_angle_deg": (-7.27, 20.85),
        "mtu_force": (0.003, 0.148),
        "length_ce": (0.682, 0.996),
        "velocity_ce": (-0.324, 1.001),
        "vnmc_torque": (0.41, 23.6),
        "m_stim": (0.01, 0.227),
        "scalefactor": (1.10, 2.58),
    },
    "l": {
        "ankle_angle_deg": (-11.79, 15.02),
        "mtu_force": (0.002, 0.218),
        "length_ce": (0.697, 1.026),
        "velocity_ce": (-0.354, 1.001),
        "vnmc_torque": (0.37, 34.8),
        "m_stim": (0.01, 0.330),
        "scalefactor": (0.74, 1.74),
    },
}
STANCE_STATE = 4  # boot_state.STANCE


def runs_of(t: np.ndarray, state: np.ndarray, value: float) -> list[tuple[float, float]]:
    """(start, end) time of each run of ``state == value``: from its first sample to the first sample after it (NaN if
    it lasts to the end of the record)."""
    on = np.concatenate([[False], state == value, [False]])
    edges = np.flatnonzero(np.diff(on.astype(int)))
    return [(t[a], t[b] if b < len(t) else np.nan) for a, b in zip(edges[::2], edges[1::2])]


def vnmc_stance_table(t: np.ndarray, state: np.ndarray, onsets: np.ndarray) -> dict[str, np.ndarray]:
    """Per complete VNMC stance (control_state 4): its duration, the number of foot-contact onsets inside it (one or more
    means it ran through swing into the next step), and where it ended on the true stride it started in (time since
    that stride's contact onset over the stride's duration; NaN outside a complete stride)."""
    durations, through, end_phase = [], [], []
    for a, b in runs_of(t, state, STANCE_STATE):
        if not np.isfinite(b):
            continue
        durations.append(b - a)
        through.append(int(np.count_nonzero((onsets > a) & (onsets < b))))
        k = np.searchsorted(onsets, a, side="right") - 1
        end_phase.append((b - onsets[k]) / (onsets[k + 1] - onsets[k]) if 0 <= k < len(onsets) - 1 else np.nan)
    return {"duration": np.array(durations), "through": np.array(through), "end_phase": np.array(end_phase)}


def _ranges(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    return (float(np.percentile(values, 1)), float(np.percentile(values, 99))) if len(values) else (np.nan, np.nan)


def analyze_vnmc(results: dict[str, list[Episode]], params, model, rate_hz: float) -> dict:
    """The VNMC's checks, per leg: sensing and the would-be torque in shadow mode, the delivered torque assisting, the
    stances (how long, how many ran through swing, where they ended), the muscle's ranges, and the gait."""
    tick = 1.0 / rate_hz
    a = {"tick_s": tick, "min_stride": params.min_stride_duration, "legs": {}, "gait": [], "nonfinite": []}

    def stances(ep, side):
        s = ep.substeps
        onsets, ends = contacts(
            s["t"], s[f"grf_{side}"], on=params.grf_on_newtons, off=params.grf_off_newtons, min_unload=params.min_unload_time
        )
        return split_contacts(onsets, ends, s["t"][-1])

    for side in SIDES:
        leg = dict(delays=[], missed=0, false=0, touches=0, loop_strides=[], cases={})
        for ep in results[VNMC_SHADOW]:
            s = ep.substeps
            stance_onsets, _, touches, undetermined = stances(ep, side)
            reported = (s["tick"] == 1) & (s[f"strike_{side}"] == 1)
            loop = s[f"strike_time_{side}"][reported]
            loop = np.array([f for f in loop if not np.any(np.abs(undetermined - f) <= tick + 1e-9)])
            delays, missed, false = strike_delays(loop, stance_onsets, max_delay=tick + 1e-9)
            leg["delays"] += list(delays)
            leg["missed"] += missed
            leg["false"] += len(false)
            leg["touches"] += len(touches)
            leg["loop_strides"] += list(np.diff(loop))
        for case in (VNMC_SHADOW, VNMC_ASSIST):
            c = dict(duration=[], through=[], end_phase=[], raw_peak=[], peak=[], impulse=[], profiles=[], share={})
            c.update(applied_error=0.0, off_tick=0, max_stance_time=0.0, ranges={})
            pooled = {key: [] for key in VNMC_RANGE_KEYS}
            states = []
            for ep in results[case]:
                s = ep.substeps
                t, state = s["t"], s[f"control_state_{side}"]
                onsets = stances(ep, side)[0]
                table = vnmc_stance_table(t, state, onsets)
                for key in ("duration", "through", "end_phase"):
                    c[key] += list(table[key])
                command = s[f"torque_nm_{side}"]
                for a_, b_ in runs_of(t, state, STANCE_STATE):
                    inside = (t >= a_) & (t < (b_ if np.isfinite(b_) else np.inf))
                    c["raw_peak"].append(float(s[f"vnmc_torque_{side}"][inside].max()))
                    c["peak"].append(float(command[inside].max()))
                dt = float(np.median(np.diff(t)))
                c["impulse"] += [float(command[(t >= x) & (t < y)].sum() * dt) for x, y in zip(onsets[:-1], onsets[1:])]
                _, rows = stride_average(t, command, onsets)
                c["profiles"] += list(rows)
                c["max_stance_time"] = max(c["max_stance_time"], float(np.nanmax(s[f"stance_time_{side}"])))
                # Once the first stance has begun: before it the muscle idles at rest and says nothing about the gait.
                walking = t >= (runs_of(t, state, STANCE_STATE) or [(np.inf, np.inf)])[0][0]
                states.append(state[walking])
                for key in VNMC_RANGE_KEYS:
                    pooled[key].append(s[f"{key}_{side}"][walking])
                if case == VNMC_ASSIST:
                    cmd, applied = s[f"cmd_{side}"], s[f"applied_{side}"]
                    c["applied_error"] = max(c["applied_error"], float(np.nanmax(np.abs(applied - cmd))))
                    c["off_tick"] += changes_off_tick(cmd, s["tick"])
            states = np.concatenate(states) if states else np.zeros(0)
            c["share"] = {k: float(np.mean(states == k)) if len(states) else np.nan for k in (1, 2, 3, 4)}
            c["ranges"] = {key: _ranges(np.concatenate(v)) if v else (np.nan, np.nan) for key, v in pooled.items()}
            leg["cases"][case] = c
        a["legs"][side] = leg
    for case, eps in results.items():
        for ep in eps:
            recorded = [ep.substeps[k] for k in ("t", "cmd_r", "cmd_l", "grf_r", "grf_l")] + list(ep.steps.values())
            if not all(np.all(np.isfinite(v)) for v in recorded) or ep.ending == "non-finite observation":
                a["nonfinite"].append((case, ep.start_index))
            strides = np.concatenate([np.diff(stances(ep, side)[0]) for side in SIDES])
            speed = (ep.steps["pelvis_tx"][-1] - ep.steps["pelvis_tx"][0]) / max(ep.duration - ep.dt, ep.dt)
            a["gait"].append(
                dict(
                    case=case,
                    index=ep.start_index,
                    duration=ep.duration,
                    ending=ep.ending,
                    speed=speed,
                    stride=float(np.mean(strides)) if len(strides) else np.nan,
                    plantarflexion=-np.degrees(min(ep.steps["ankle_angle_r"].min(), ep.steps["ankle_angle_l"].min())),
                )
            )
    a["same_as_exo_off"] = max(
        (
            float(np.max(np.abs(off.steps[j] - on.steps[j]))) if len(off.steps[j]) == len(on.steps[j]) else np.inf
            for off, on in zip(results[EXO_OFF], results[VNMC_SHADOW])
            for j in STEP_JOINTS
        ),
        default=np.nan,
    )
    return a


def plot_vnmc(results: dict[str, list[Episode]], analysis: dict, params, model, path: pathlib.Path) -> None:
    """Per leg, the torque over the true stride: what the VNMC would command on the unassisted gait (shadow) and what
    it delivers assisting; and 3 s of the right leg assisting: command, raw muscle torque, state and ankle angle."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 9), layout="constrained")
    grid = fig.add_gridspec(4, 2, height_ratios=[1.3, 1.0, 0.45, 0.8])
    for col, side in enumerate(SIDES):
        ax = fig.add_subplot(grid[0, col])
        for case, color in ((VNMC_SHADOW, REFERENCE), (VNMC_ASSIST, DELIVERED)):
            rows = np.array(analysis["legs"][side]["cases"][case]["profiles"])
            if not len(rows):
                continue
            phase = (np.arange(rows.shape[1]) + 0.5) / rows.shape[1]
            mean, sd = np.nanmean(rows, axis=0), np.nanstd(rows, axis=0)
            ax.fill_between(phase, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
            label = "would command, unassisted gait" if case == VNMC_SHADOW else "delivered, assisting"
            ax.plot(phase, mean, color=color, linewidth=2, label=f"{label} ({len(rows)} strides)")
        ax.set(title=f"{'Right' if side == 'r' else 'Left'} ankle", xlabel="gait phase (contact to contact)")
        ax.set_ylabel("plantarflexion torque (N·m)")
        ax.set_xlim(0, 1)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    eps = results[VNMC_ASSIST]
    if eps:
        ep = max(eps, key=lambda e: e.duration)
        s = ep.substeps
        t0 = s["t"][0] + min(4.0, max(ep.duration - 3.0, 0.0))
        window = (s["t"] >= t0) & (s["t"] <= t0 + 3.0)
        ax = fig.add_subplot(grid[1, :])
        ax.plot(s["t"][window], s["vnmc_torque_r"][window], color=REFERENCE, linewidth=1.5, label="raw muscle torque")
        ax.step(s["t"][window], s["cmd_r"][window], where="post", color=DELIVERED, linewidth=2, label="command (applied)")
        ax.set(title=f"Right ankle, 3 s of episode {ep.start_index}", ylabel="torque (N·m)")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        ax2 = fig.add_subplot(grid[2, :], sharex=ax)
        ax2.step(s["t"][window], s["control_state_r"][window], where="post", color=INK, linewidth=1.5)
        ax2.set(ylabel="state", yticks=[1, 2, 3, 4], yticklabels=["reel-out", "swing", "reel-in", "stance"])
        ax3 = fig.add_subplot(grid[3, :], sharex=ax)
        ax3.plot(s["t"][window], s["ankle_angle_deg_r"][window], color=INK, linewidth=1.5)
        ax3.set(ylabel="ankle angle, the boot's\n(deg, plantarflexion +)", xlabel="time (s)")
    for ax in fig.axes:
        ax.grid(color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def report_vnmc(analysis: dict, *, policy, config_path, sweep: list[tuple[int, float]], kept: list[int], rate_hz: float) -> str:
    legs = analysis["legs"]

    def cell(case, key, scale=1.0, digits=2, unit=""):
        return " | ".join(_fmt(legs[s]["cases"][case][key], scale, digits, unit) for s in SIDES)

    lines = [
        "# VNMC rollouts",
        "",
        f"Policy `{policy}`, config `{config_path}`, {datetime.date.today().isoformat()}.",
        "",
        f"{len(sweep)} start indices tried with the exo off; {len(kept)} walked long enough and were run in every case: "
        + ", ".join(f"{i} ({d:.1f} s)" for i, d in sweep if i in kept)
        + ".",
        "",
        "## Pass bar",
        "",
        "| check | right | left | |",
        "|---|---|---|---|",
        f"| no NaN or crash | | | {_pass(not analysis['nonfinite'])} |",
    ]

    def row(name, fmt, ok):
        cells = [fmt(legs[side]) for side in SIDES]
        lines.append(f"| {name} | {cells[0]} | {cells[1]} | {_pass(all(ok(legs[side]) for side in SIDES))} |")

    row(
        f"sensing: every stance (a contact of {MIN_STANCE * 1e3:.0f} ms or more) a strike, dated within a tick",
        lambda leg: (
            f"{len(leg['delays'])} of {len(leg['delays']) + leg['missed']}; delay {_fmt(leg['delays'], 1e3, unit=' ms')}"
        ),
        lambda leg: leg["missed"] == 0 and len(leg["delays"]) > 0,
    )
    row(
        "sensing: no other strike, such as a scuff in swing",
        lambda leg: f"{leg['false']} ({leg['touches']} shorter touches)",
        lambda leg: leg["false"] == 0,
    )
    same = analysis["same_as_exo_off"]
    lines.append(f"| shadow gait identical to exo off | max difference {same:.1e} | | {_pass(same == 0)} |")
    row(
        "applied torque = command",
        lambda leg: f"max error {leg['cases'][VNMC_ASSIST]['applied_error']:.1e} N·m",
        lambda leg: leg["cases"][VNMC_ASSIST]["applied_error"] < 1e-6,
    )
    row(
        "torque held between ticks",
        lambda leg: f"{leg['cases'][VNMC_ASSIST]['off_tick']} changes off a tick",
        lambda leg: leg["cases"][VNMC_ASSIST]["off_tick"] == 0,
    )
    lines += [
        "",
        "## Stances and torque",
        "",
        "Per stance (control state 4) or per stride between foot-contact onsets. A stance that ran through swing is one "
        "with a contact onset inside it: its raw muscle torque never passed 6.25 N·m (5 N·m filtered), or never rose again below 80% of its "
        "peak, so its toe-off fired late.",
        "",
        "| | shadow: right | shadow: left | assisting: right | assisting: left |",
        "|---|---|---|---|---|",
    ]

    def both(key, scale=1.0, digits=2, unit=""):
        return f"{cell(VNMC_SHADOW, key, scale, digits, unit)} | {cell(VNMC_ASSIST, key, scale, digits, unit)}"

    lines += [
        f"| stance duration (s) | {both('duration')} |",
        f"| stance ends at phase (of its true stride) | {both('end_phase', digits=3)} |",
        "| stances through swing | "
        + " | ".join(
            f"{int(np.sum(np.array(legs[s]['cases'][c]['through']) > 0))} of {len(legs[s]['cases'][c]['through'])}"
            for c in (VNMC_SHADOW, VNMC_ASSIST)
            for s in SIDES
        )
        + " |",
        "| longest stance_time (s) | "
        + " | ".join(f"{legs[s]['cases'][c]['max_stance_time']:.2f}" for c in (VNMC_SHADOW, VNMC_ASSIST) for s in SIDES)
        + " |",
        f"| raw muscle peak per stance (N·m) | {both('raw_peak', digits=1)} |",
        f"| command peak per stance (N·m) | {both('peak', digits=1)} |",
        f"| impulse per stride (N·m·s) | {both('impulse')} |",
        "| share of time in reel-out / swing / reel-in / stance | "
        + " | ".join(
            " / ".join(f"{100 * legs[s]['cases'][c]['share'][k]:.0f}%" for k in (1, 2, 3, 4))
            for c in (VNMC_SHADOW, VNMC_ASSIST)
            for s in SIDES
        )
        + " |",
        "",
        "## The muscle's ranges against the boot's",
        "",
        "p1 to p99 over every substep from the first stance on, against the same leg's in the VNMC sessions' logs (every "
        "row while walking). The muscle follows the absolute ankle angle (the boot's, through each side's standing "
        "angle), so where the policy's ankle range differs from the participant's the muscle works elsewhere.",
        "",
        "| | right: logs | right: shadow | right: assisting | left: logs | left: shadow | left: assisting |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in VNMC_RANGE_KEYS:
        cells = []
        for s in SIDES:
            cells.append("{:.3g} to {:.3g}".format(*VNMC_LOG_RANGES[s][key]))
            cells += ["{:.3g} to {:.3g}".format(*legs[s]["cases"][c]["ranges"][key]) for c in (VNMC_SHADOW, VNMC_ASSIST)]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Gait",
        "",
        "Reported, not required: the policy was not trained with this assistance.",
        "",
        "| case | start index | duration (s) | ending | speed (m/s) | stride (s) | peak plantarflexion (°) |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in analysis["gait"]:
        lines.append(
            f"| {g['case']} | {g['index']} | {g['duration']:.1f} | {g['ending']} | {g['speed']:.2f} | {g['stride']:.2f} "
            f"| {g['plantarflexion']:.1f} |"
        )
    return "\n".join(lines) + "\n"


# --- the controllers' suites ----------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Suite:
    """One controller's rollouts: its device-env config, its cases, and its checks.

    Every case runs in its own env from the start indices the exo-off sweep keeps, after the exo-off case itself. A
    case is the ``device_controller`` and any ``exo_controller_params`` it overrides. ``extra_keys`` are
    ``diagnostics()`` keys the ``Recorder`` records beyond its own fields.
    """

    config: pathlib.Path
    cases: dict[str, dict]
    analyze: Callable  # (results by case, params, model, rate_hz) -> analysis
    plot: Callable  # (results, analysis, params, model, path to write)
    report: Callable  # (analysis, *, policy, config_path, sweep, kept, rate_hz) -> markdown
    extra_keys: tuple[str, ...] = ()


SUITES = {
    "4PTS": Suite(
        config=DEFAULT_CONFIG,
        cases={
            "4PTS sensing": dict(device_controller="exoboot_spline", peak_torque=0.0, bias_torque=0.0),
            "4PTS": dict(device_controller="exoboot_spline"),
        },
        analyze=analyze,
        plot=plot,
        report=report,
    ),
    "VNMC": Suite(
        config=VNMC_CONFIG,
        cases={
            VNMC_SHADOW: dict(device_controller="exoboot_vnmc", shadow_mode=True),
            VNMC_ASSIST: dict(device_controller="exoboot_vnmc"),
        },
        analyze=analyze_vnmc,
        plot=plot_vnmc,
        report=report_vnmc,
        extra_keys=VNMC_EXTRA_KEYS,
    ),
}
# Every case by name, the exo-off one first.
CASES = {EXO_OFF: EXO_OFF_CASE, **{name: case for suite in SUITES.values() for name, case in suite.cases.items()}}


def suite_of(case: str) -> Suite | None:
    """The suite a case belongs to; None for the exo-off case, which they all share."""
    return next((suite for suite in SUITES.values() if case in suite.cases), None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("policy", type=pathlib.Path, help="a trained policy's .zip")
    parser.add_argument("--controller", choices=list(SUITES), default="4PTS", help="whose cases and checks to run")
    parser.add_argument("--config", type=pathlib.Path, default=None, help="its device-env config (default: its own)")
    parser.add_argument("--min-seconds", type=float, default=8.0, help="keep start indices that walk at least this long")
    parser.add_argument("--max-kept", type=int, default=10, help="at most this many, the longest")
    parser.add_argument("--index-step", type=int, default=20, help="try every n-th start index (30 Hz reference)")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)
    suite = SUITES[args.controller]
    config_path = args.config or suite.config
    out = args.out or pathlib.Path("rl_train/results/rollouts") / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    policy = load_policy(args.policy)
    results: dict[str, list[Episode]] = {}
    env, config = make_env(config_path, EXO_OFF_CASE)
    if policy.observation_space.shape != env.observation_space.shape:
        raise ValueError(
            f"the policy takes observations of shape {policy.observation_space.shape}, the config's env gives "
            f"{env.observation_space.shape}: use a config with the policy's own observation layout"
        )
    run = dict(max_steps=config.env_params.custom_max_episode_steps, safe_height=config.env_params.safe_height)
    # The sweep keeps only each start's duration: episodes are reproducible, so the kept ones are simply run again.
    starts = range(0, int(env._reference_data_length * 0.8), args.index_step)
    sweep = []
    for n, index in enumerate(starts, 1):
        ep = run_episode(env, policy, env.device_controller, index, case=EXO_OFF, **run)
        sweep.append((index, ep.duration))
        print(f"exo off, start {index} ({n}/{len(starts)}): {ep.duration:.1f} s, {ep.ending}", flush=True)
    long_enough = sorted((d, i) for i, d in sweep if d >= args.min_seconds)[::-1][: args.max_kept]
    kept = sorted(i for _, i in long_enough)
    print(f"{len(kept)} of {len(sweep)} start indices kept (at least {args.min_seconds:g} s): {kept}", flush=True)
    if not kept:
        raise SystemExit(f"no episode walked long enough to test {args.controller}; lower --min-seconds or use another policy")
    for case in (EXO_OFF, *suite.cases):
        if case != EXO_OFF:
            env.close()
            env, config = make_env(config_path, suite.cases[case], suite.extra_keys)
        results[case] = [run_episode(env, policy, env.device_controller, i, case=case, **run) for i in kept]
        print(f"{case}: " + ", ".join(f"{e.start_index} {e.duration:.1f} s {e.ending}" for e in results[case]), flush=True)

    # The last case's env and parameters, kept open for the analysis: it may rebuild the controller on this model.
    model = getattr(env.sim.model, "ptr", env.sim.model)
    params = config.env_params.exo_controller_params
    rate_hz = params.controller_rate_hz
    analysis = suite.analyze(results, params, model, rate_hz)
    suite.plot(results, analysis, params, model, out / "torque_vs_phase.png")
    env.close()
    (out / "report.md").write_text(
        suite.report(analysis, policy=args.policy, config_path=config_path, sweep=sweep, kept=kept, rate_hz=rate_hz),
        encoding="utf-8",
    )
    save_episodes(results, out / "episodes.npz")
    print((out / "report.md").read_text(encoding="utf-8"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
