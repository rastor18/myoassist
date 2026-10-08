# ExoBoot VNMC — a scripted exo, the policy learns muscles only

The NeuMove ExoBoot's virtual neuromuscular controller (`VirtualNeuroMuscularController`, the boot's config
`vnmc_config.py`), ported to run inside the RL env in place of the policy's exo torque. One Geyer-type
muscle-tendon unit about each ankle is driven by the measured ankle angle and stimulated by positive force feedback;
its torque is scaled stance by stance, and it decides its own toe-off. It runs inside the physics loop at 150 Hz, close
to the boot's own 175 Hz loop. The policy still emits the two exo actions; the controller overrides them on every
physics substep. Code: `myoassist_utils/exo_ctrl/vnmc.py`, on the seam of `rl_train/envs/device_control.py`.

Two configs on `DephyExoBoot_L1`, the exo-off experiment but for the controller:

| config | exo |
|---|---|
| `imitation_22_DephyExoBoot_L1_exoboot_vnmc.json` | the VNMC at 150 Hz, inside the 1200 Hz physics |
| `imitation_22_DephyExoBoot_L1_exoboot_vnmc_shadow.json` | the VNMC running on every substep but applying no torque (`shadow_mode`) |

The exo-off baseline is 4PTS's: `../exoboot_spline/imitation_22_DephyExoBoot_L1_exo_off.json`. Everything outside
`exo_controller_params` is the same in all three (`tools/tests/test_vnmc_device_env.py` checks it).

## Getting started

The code is on the `feat/exoboot-vnmc` branch, on top of the four-point spline's (`exoboot_spline/README.md`). Install
as the repo's own README says (`uv pip install -e .`), from a clone of that branch. Then, from the repo root, run the
tests; the VNMC's are `tools/tests/test_exoboot_vnmc.py` (the controller), `test_vnmc_device_env.py` (in the env and
its configs), `test_replay_vnmc_session.py` (the log replay) and the VNMC cases of `test_rollout_controllers.py` and
`test_render_controller_video.py`:

```bash
python -m pytest tools/tests -q
```

From there: train with a config (below), check the VNMC on a trained policy with `tools/rollout_controllers.py
--controller VNMC` ([Tested on a trained policy](#tested-on-a-trained-policy)), or replay a boot session log through the
port with `tools/replay_vnmc_session.py` ([Validated against the boot](#validated-against-the-boot)).

## Train

Train the baseline first, then check the VNMC on its rollouts (shadow mode first: it shows what the VNMC would do on
that gait without changing it):

```bash
python rl_train/run_train.py --config_file_path rl_train/train/train_configs/exoboot_spline/imitation_22_DephyExoBoot_L1_exo_off.json
python rl_train/run_train.py --config_file_path rl_train/train/train_configs/exoboot_vnmc/imitation_22_DephyExoBoot_L1_exoboot_vnmc.json
```

Every controller field is overridable from the command line as `--config.env_params.exo_controller_params.<field>
<value>`. Evaluation and `exo_phase_valid` work as for 4PTS (`exoboot_spline/README.md`, "Evaluate"); for the VNMC,
`exo_phase_valid` is the share of steps with a gait phase, which gates reel-in and so the next stance.

## What the controller does

As the boot ran it in the VNMC sessions. The boot's committed code is an earlier snapshot; where its logs and that code
disagree, the port follows the logs (the torque filter and the toe-off, below).

* **The muscle** (`MusculoTendonJoint`, the boot's `muscle_model.MusculoTendonJoint`): F_max 4000 N, l_opt 0.04 m,
  v_max 6 l_opt/s, l_slack 0.26 m, rho 0.5, e_ref 0.1, a 0.04 m moment arm, phi_ref 0, activation time constants 10 ms
  rising and 40 ms falling, one explicit Euler step per controller tick. Its length follows the ankle angle as the
  boot's encoder reads it, in degrees, plantarflexion-positive: dorsiflexion stretches it. It steps on every tick in
  every state, and is reset only with the episode, so its state at a heel strike depends on the swing before it.
* **The reflex**: in stance the stimulation is `0.01 + vnmc_gain × F/F_max` (`vnmc_gain` 1.468, the boot's
  `VNMC_GAIN`), with the force from 20 ms earlier: 3 ticks at 150 Hz, 4 at the boot's 175 Hz (22.9 ms, as
  `round(0.02 × 175)` makes it). Outside stance it is 0.01.
* **The torque filter, as it ran**: the boot passes the muscle's torque through an exponential moving average
  (`FILT_ALPHA` 0.8, meant to keep ankle noise from firing the toe-off early), but its `update_muscle_model` resets the
  average to 0 just before each update, so the "filtered" torque is 0.8 × the muscle's, with no smoothing. The peak,
  the toe-off and the scaling all run on it, here too.
* **The scaling**: each stance's command is `min(peak_torque, scalefactor × filtered torque)`, with
  `scalefactor = peak_torque / the previous stance's filtered peak`, and `peak_torque / 100` before the first toe-off.
  So a stance whose torque reaches the previous stance's peak commands `peak_torque` (25 N·m), and is clipped there;
  the filter's 0.8 cancels, except in the first stance. The committed code has no filter and a hard-coded 25 N·m; the
  logs show the 0.8 and the live `PEAK_TORQUE` (it followed the session's warm-up ramp).
* **Toe-off from the muscle's own torque**: once the stance's filtered peak passes 5 N·m, eight ticks in a row with the
  torque not rising latch it; it fires on the first later tick at most 80% of the peak and not below the tick before.
  (The committed code waits four ticks; the logs show eight.) The state machine sees it one tick later (the toe-off
  tick itself still commands torque), and reels out.
* **The state machine around it** is the boot's WALKING task (`BootStateMachine`): reel-out (0.2 s, the boot's timer:
  `SoftReelOutController` forced to complete on time) → swing → a heel strike with a gait phase → reel-in
  (`reel_in_time`) → stance → the VNMC's toe-off → reel-out. Only stance applies torque. **Nothing but the VNMC's
  toe-off ends its stance**: not a heel strike, not a lost gait phase. A stance whose filtered torque never passes 5 N·m
  (6.25 N·m of the muscle's), or never rises again below 80% of its peak, runs through swing and the next step, until a later rise fires it;
  `diagnostics()["stance_time"]` shows it.
* **Heel strikes from foot contact force**, as for 4PTS (the boot uses a shank gyro): a contact counts once it has
  lasted `min_contact_time` (50 ms), dated back to its onset, and ends once the foot has been unloaded for
  `min_unload_time`. The stride-average gait phase they give gates reel-in: no reel-in before the third heel strike.
* **The ankle angle from the boot's encoder** (`AnkleEncoder`): encoder clicks plus a per-side offset that puts the
  standing keyframe at `ankle_standing_angle_r_deg` / `_l_deg` (−1.85 / −8.47°): what the boot read in the quiet
  standing at the start of the VNMC sessions (mean of both sessions). The muscle's working point is the absolute
  angle, so these matter more here than for the DL task.
* **In the physics loop at 150 Hz** (`controller_rate_hz`): every 8th substep, its torque held in between, as for 4PTS.
  A step costs 1.21x a stock step (4PTS: 1.18x; the in-loop seam alone: 1.16x), single env, pinned to one core
  (`tools/bench_device_step.py --cpu 3`).
* **It reports its state as the boot logs it.** Each leg's `diagnostics()` gives the common keys (`torque_nm`, `phase`,
  `phase_valid`, `in_stance`, `heel_strike`, `strike_time`, `control_state` in the boot's codes 1 reel-out, 2 swing, 3
  reel-in, 4 stance, `stride_estimate`) and the VNMC's own: `mtu_force`, `length_ce`, `velocity_ce` (normalized, as the
  boot logs them), `vnmc_torque` (the raw muscle torque), `m_stim`, `scalefactor`, `stance_time` and
  `ankle_angle_deg`.

## When it applies torque, and how to change it

All fields of `exo_controller_params`.

1. **Once per episode: the warm-up.** A heel strike starts reel-in, then stance, but the boot's state machine also
   requires a gait phase (`state_machines.py:196-198`), which takes two whole strides (`num_strides_required` 2): so
   the first two strikes of an episode start nothing, and the third does, as for 4PTS. On the boot this gate passed
   unseen: its sessions began with `SWING_ONLY` on, and by the time it was switched off (15 s in) the phase was valid,
   so the first heel strike after that started the first stance. Then the first stance is scaled by `peak_torque / 100`: it commands a
   fifth of the muscle's torque (`0.25 × 0.8`, the filter's 0.8). From the second on, each stance is scaled to the one before. The
   toe-off tick already uses the new scaling, so the first stance ends with a one-tick spike (6.7 ms) to about four
   times its torque, as on the boot.
2. **Every stride: reel-in.** No torque for `reel_in_time` (0.162 s) after each heel strike. The boot ends reel-in on
   cable slack, or after 0.2 s; in the VNMC sessions it took 152–175 ms (the mean of each leg in each session), ± 23–26
   ms stride to stride.
3. **Every stride: the stance ends at the VNMC's toe-off**, which depends on the muscle, so on the ankle angle and on
   `vnmc_gain`. A higher gain drives the reflex harder: more force, a later peak. The ankle's standing angles shift
   the muscle's length, and so its force at a given joint angle.
4. **The size**: `peak_torque` (25 N·m) is both the clip and the level each stance is scaled to reach.

## Validated against the boot

`tools/replay_vnmc_session.py` replays a VNMC session log through the port, one layer at a time, using the boot's own
logged inputs:

```bash
python tools/replay_vnmc_session.py "<session dir>/<date>_<time>_<subject>_<trial>_" --start 31 --end 485
```

On one participant's two VNMC sessions, over the walking at constant parameters (31–485 s, after the warm-up ramp of
`PEAK_TORQUE`; 385–386 strides per leg per session), at the boot's 175 Hz:

| layer | input | result |
|---|---|---|
| muscle | logged ankle angle and stimulation | bit for bit on 99.9% of rows (99.5% for the velocity); the rest within 2×10⁻¹² N·m, rounding |
| reflex | the port's own muscle, logged states | stimulation bit for bit on 99.8–99.97% of rows, the rest within 5×10⁻¹⁴ |
| scaling | the same | scalefactor and command in stance bit for bit on 99.5–100% of rows, the rest within 6×10⁻¹² |
| toe-off | the same | the port's toe-off fires on the boot's last stance row in 100% of stances |
| whole leg | logged heel strikes and angle, the boot's own reel-in ends | control state agrees on 99.9% of rows; stance ends on the boot's row in 99.7–100% of strides |
| | the same with a fixed reel-in (each leg's mean) | control state agrees on 98.6–98.7% of rows; stance ends on the boot's row in 93.5–95.6% of strides |
| impulse | the same | per stride within 0.5% of the boot's command (8.9–9.7 N·m·s) |
| 150 Hz in-loop | the device env's ticks, the logged angle read at each tick | per stride: lag +2.1 to +5.8 ms, peak within 0.4%, median impulse within 0.8%; stance ends 5.5–6.7 ms later; 5–6 of 385 strides off by more than 10% |

* **Iterations the boot did not log.** It writes a row only when its actpack has a new packet, but runs the muscle on
  every loop iteration. At 25–62 of 71–105 gaps per log, the muscle shows one to four iterations the log does not
  have (35–69 per log); the replay finds them by trying each count, and steps through them. Without them the torque
  is off by up to 4 N·m after each one. On the boot the reflex and the toe-off ran on those iterations too: a toe-off
  can fire on one (5 of the 1546 stances did), and the log then shows reel-out on the next row.
* **Eight falling ticks, not four.** With the committed code's four, two stances of the 1546 toe off 12 and 61 rows
  before the boot did, and their command and the next stance's scaling are off by up to 3 N·m; with eight every
  stance ends on the boot's row. The two rules differ only on a stance whose torque falls for fewer than eight ticks
  before a rise below 80% of its peak.
* **Reel-in** is where a fixed time costs: the boot's ended on cable slack, ± 23–26 ms; a stance that starts earlier or
  later builds its reflex differently and can end a few ticks off. It costs nothing in impulse (within 0.5%).
* **At 150 Hz** the 20 ms delay is exactly 3 ticks (22.9 ms at 175 Hz) and the Euler step is 6.7 ms: the raw muscle
  peaks higher, which the per-stance scaling takes out (the command's peak is within 0.4%). The stance ends about
  a tick later on average. The few strides off by more than 10% are ones whose toe-off fires on a different rise of
  the torque, or a stride later: the rule's "first rise below 80%" is sensitive to small wiggles near the threshold.
  (The mean lag of +5.8 ms is one leg's, pulled up by such a stride; the others are +2.1 to +4.0.)
  The raw peak is 0.5–2.6% higher. The same replay at 175 Hz in the loop (the tick grid alone) lags +1.5 to +2.1 ms,
  impulse within 0.4%.

Re-run the replay for each boot pair or session before relying on `reel_in_time` or the standing angles; it prints
both (`exo_controller_params: ...`).

Checked in the env (`tools/tests/test_vnmc_device_env.py`): in shadow mode the device env reproduces the stock exo-off
env bit for bit (qpos, qvel, act, observation, reward and done on every one of 120 steps); each tick reads the leg's foot
force and its ankle angle in the boot's degrees (the standing angle at the keyframe); the VNMC's command is the torque
the ankle feels; and every reset starts the controller over (reel-out, the muscle at rest, the scaling at 25/100).

## Tested on a trained policy

`tools/rollout_controllers.py --controller VNMC` runs a trained policy in the device env three times from the same
start points in the reference motion: with the exo off, with the VNMC in shadow mode (running, applying nothing), and
with the VNMC. It records every physics substep, the VNMC's own diagnostics included, and checks its strikes against
the policy's foot contacts, the applied torque against the command, and its stances:

```bash
python tools/rollout_controllers.py <train_session_...>/trained_models/<model>.zip --controller VNMC --index-step 40 --max-kept 6
```

It writes `report.md`, `torque_vs_phase.png` and `episodes.npz` to `rl_train/results/rollouts/<time>/`. On the MyoAssist
tutorial policy (the one 4PTS was tested on: `train_session_20250728-161129_tutorial_partial_obs`, `model_19939328.zip`),
which walks on `DephyExoBoot_L1` with the exo off for 32–33 s at 1.10–1.12 m/s, strides of 1.11 s (the 6 start points of
54 tried that walk at least 8 s; 182 and 178 strikes):

| check | result |
|---|---|
| strikes (shadow) | every stance a strike (182 of 182 right, 178 of 178 left), dated +2.9 to +3.1 ± 1.8–1.9 ms after the contact began; no other strike (the right foot's 13 toe scuffs ignored) |
| shadow | the gait is the exo-off gait, bit for bit |
| what it would do (shadow) | stance from phase 0.15 (reel-in after the foot lands) for 0.50–0.52 ± 0.02 s, ending at phase 0.60–0.62 ± 0.01 of the true stride; none ran through swing (0 of 330); raw muscle peak 38–42 N·m, command peak 23 N·m, 5.4–5.5 N·m·s per stride |
| applied torque | equals the command (to 2×10⁻¹⁵ N·m), changing only on the 150 Hz ticks |
| assisting | stance 0.45–0.52 ± 0.05 s, ending at phase 0.59–0.66 ± 0.04–0.06; none ran through swing (0 of 61); raw peak 32–34 N·m, command peak 18–21 N·m |

**The command's shape is not the boot's,** and that is the policy's gait, not the port. Per stride (the boot's two
sessions against the sim's shadow runs), stance starts at the same phase (0.13–0.15 on the boot, 0.15 here), and the
muscle's force, stimulation and raw torque follow the boot's through the first third of the stride. Then they part:
the boot's command ramps steadily to a peak at 0.51–0.53 of the stride and its stance ends at 0.67; the sim's rises
late and steeply to a narrower peak at 0.43–0.47, and its stance ends at 0.60–0.62. The muscle's length follows the
ankle angle, and its positive force feedback grows whenever the ankle dorsiflexes and stretches it. The participant's
ankle stays about level through mid-stance and pushes off from 0.6 of the stride; the policy's dorsiflexes 13–20° more
through mid-stance and pushes off about 0.1 of the stride earlier. A constant offset is not the cause: read 7° more
plantarflexed, the sim's ankle gives the same command (the per-stance scaling absorbs the force level). A policy
trained with the VNMC walks differently, so its shape should move, but that has not been tested.

**The muscle works off the participant's range.** Through the standing angles, the policy's ankle reads as the boot's
would, and over its walking it spans −21 to +12° (right) and −29 to +8° (left), p1 to p99, against −7 to +21° and −12 to
+15° on the boot: 13–17° more dorsiflexed. The muscle's force and torque run higher than on the boot (force up to
0.25 and 0.28 of F_max against 0.15 and 0.22; raw torque up to 40 and 45 N·m against 24 and 35), which the per-stance
scaling normalizes to `peak_torque`, and the contractile element sits longer (0.72–1.05 l_opt against 0.68–1.03), so
the toe-off and the shape follow this gait, not the participant's.

**Gait survival (reported, not required).** With the VNMC's torque, which the policy was never trained with, it drifts
off the reference motion (the imitation env's 0.6 rad termination) after 5.5–11.9 s from all 6 start points (one
fall), at 1.09–1.15 m/s with strides shortening to 0.99–1.08 s; 4PTS lasted 5–14 s on the same policy. Most of those endings
are the imitation env's drift stop, not falls, and the torque is what the policy cannot take. From the same 6 start
points, with the policy unchanged:

| `peak_torque` | with the drift stop | without it (a fall or the 33.3 s limit ends it) |
|---|---|---|
| 0 (exo off) | 32.3–33.3 s | 33.3 s, all 6 |
| 10 N·m | 8.4–20.5 s, all off the reference | 33.3 s, all 6 |
| 15 N·m | 8.3–20.5 s, all off the reference | 33.3 s at 5 of 6; one fall at 14.5 s |
| 25 N·m | 5.5–11.9 s (one fall) | 5.6–12.3 s, all falls |

So for evaluating the VNMC on a policy trained without it, 10 N·m with the drift stop off
(`env._out_of_trajectory_threshold = inf`, as `tools/render_controller_video.py` sets it) keeps it walking the whole
episode at 1.10–1.11 m/s, and 15 N·m nearly always does. A policy trained with the VNMC should take 25 N·m, but that has not been tested.

To see it, `tools/render_controller_video.py --case VNMC` renders an episode with, per leg, the VNMC's muscle: its
stimulation, force and length, and its raw torque against this stance's peak, the 80% the toe-off must fall below,
and the previous stance's peak the command is scaled to:

```bash
python tools/render_controller_video.py <train_session_...>/trained_models/<model>.zip --case VNMC --start 1280 --seconds 8
```

## Known limits

* **The muscle follows the absolute ankle angle.** The standing angles map the model's ankle onto the boot's, and the
  muscle's length, force and so its toe-off follow from there. They are one participant's, read in quiet standing; a
  policy whose ankle works in another range drives the muscle elsewhere ([Tested on a trained
  policy](#tested-on-a-trained-policy) compares the ranges).
* **A stance can run through swing.** If the muscle's torque never passes 6.25 N·m in a stance (5 N·m filtered), or never rises again below
  80% of its peak, the stance does not end at push-off; it lasts until a later rise fires the toe-off, assisting
  through swing. `stance_time` in `diagnostics()` shows it; the rollout report counts such stances.
* **The scaling adapts to whatever gait the policy walks**, stance by stance: it normalizes the muscle's peak to
  `peak_torque`, which is the point, but it also makes co-adaptation likelier than with a fixed profile.
* **Reel-in is a fixed time** in place of the boot's cable slack (± 23–26 ms on the boot).
* **The simulated exo is an ideal torque source.** By its motor current, the boot delivered 1–2% more impulse than it
  commanded (cable tension in reel-in, a transient at reel-out).
