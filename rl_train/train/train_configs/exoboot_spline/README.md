# ExoBoot four-point spline (4PTS) — a scripted exo, the policy learns muscles only

The NeuMove ExoBoot's `FourPointSplineController`, ported to run inside the RL env in place of the policy's exo
torque, with the boot's tuned hardware parameters. It runs inside the physics loop at 150 Hz, close to the boot's own
175 Hz loop. The policy still emits the two exo actions; the controller overrides them on every physics substep.
Code: `myoassist_utils/exo_ctrl/` and `rl_train/envs/device_control.py`.

Two configs on `DephyExoBoot_L1`, the same experiment but for the controller:

| config | exo |
|---|---|
| `imitation_22_DephyExoBoot_L1_exo_off.json` | pinned to zero torque — the baseline |
| `imitation_22_DephyExoBoot_L1_exoboot_spline.json` | 4PTS at 150 Hz, inside the 1200 Hz physics |

## Getting started

The code is on the `feat/exoboot-4pts` branch of `neumovelab/myoassist`. Install as the repo's own README says
(`uv pip install -e .`), from a clone of that branch:

```bash
git clone -b feat/exoboot-4pts https://github.com/neumovelab/myoassist.git
```

Then, from the repo root, run the tests (about 4 minutes; the 4PTS ones are `tools/tests/test_exoboot_spline.py`,
`test_device_control.py`, `test_replay_exoboot_log.py` and `test_rollout_controllers.py`):

```bash
python -m pytest tools/tests -q
```

From there: train with a config (below), check 4PTS on any trained policy with `tools/rollout_controllers.py`
([Tested on a trained policy](#tested-on-a-trained-policy)), or replay a boot session log through the port with
`tools/replay_exoboot_log.py` ([Validated against the boot](#validated-against-the-boot)).

## Train

Train the baseline first, then check 4PTS on its rollouts with `tools/rollout_controllers.py`: the heel-strike
thresholds were set on one policy's gait (below).

```bash
python rl_train/run_train.py --config_file_path rl_train/train/train_configs/exoboot_spline/imitation_22_DephyExoBoot_L1_exo_off.json
python rl_train/run_train.py --config_file_path rl_train/train/train_configs/exoboot_spline/imitation_22_DephyExoBoot_L1_exoboot_spline.json
```

Every controller field is overridable from the command line. The boot also runs a 3 N·m bias torque, which keeps its
Bowden cable taut; the simulation has no cable, so the bias is off here (it would add 20% to the impulse per stride).
To match what the boot delivered:

```bash
python rl_train/run_train.py --config_file_path rl_train/train/train_configs/exoboot_spline/imitation_22_DephyExoBoot_L1_exoboot_spline.json \
  --config.env_params.exo_controller_params.bias_torque 3
```

## Evaluate

```bash
python -m rl_train.run_policy_eval rl_train/results/<train_session_...> --no-show --steps 1000 --regen
python tools/score_exo_policy.py    rl_train/results/<train_session_...>/analyze_results_00
python tools/plot_kinematics_exo.py rl_train/results/<train_session_...>/analyze_results_00 -o out.png
```

Evaluation rebuilds the env from the run's `session_config.json`. Until 2026-10-01 both `run_policy_eval.py` and the
in-training analyzer read it as the base imitation config, which silently dropped `exo_controller_params`: evaluation
rollouts of a scripted-exo run had **no scripted exo**. Regenerate (`--regen`) any evaluation made before then.

`train_log.json` carries `exo_phase_valid` in `average_reward_dict_per_episode` (weight 0, so it is not a reward).
**Divide it by `average_num_timestep`** to get the fraction of steps the exo was actually assisting. Read it
before reading anything else: until gait is steady the controller has no phase and applies no torque, which is
most likely early in training, so a low fraction means the "assisted" run was largely unassisted.

## What the controller does

As on the boot: heel strikes set a stride-average gait phase, and the spline gives torque over that phase.

* **Stance only.** On the boot the spline is the stance controller of a four-state machine:
  reel-in → stance → reel-out → swing. The other three manage cable slack and apply no assist, so torque is
  zero outside stance.
  * **Stance starts `reel_in_time` after the heel strike** (default 0.157 s). The boot first reels in its
    cable, under voltage control, until the slack falls below a cutoff or for 0.2 s at most. That timeout is
    hard-coded in the boot's `SmoothReelInController`; the `REEL_IN_TIMEOUT` it is passed is not used. On the
    validation log reel-in took 150 / 164 ms (left / right) at 1.25 m/s, so it usually ended on the slack, before the
    timeout. There is no cable here, so a fixed time stands in. It is a time, not a phase: across speed changes
    (75–127 steps/min) it did not follow step frequency.
  * **Stance ends at `toe_off_fraction`** (the boot's `TOE_OFF_FRACTION`, 0.60). With the tuned knots that cuts
    torque mid-fall, from 9.5 N·m (11.3 with the boot's bias), since `fall_fraction` (0.641) lies past toe-off.
* **No torque until gait is steady**: the third heel strike, i.e. two whole strides inside [0.6, 2.0] s
  ([When the assistance starts](#when-the-assistance-starts-and-how-to-change-it)).
* **It reports its state as the boot logs it.** Each leg's `diagnostics()` gives `control_state` in the boot's codes:
  2 swing (also before a gait phase exists, as on the boot, which only enters reel-in at a heel strike once it has
  one), 3 reel-in, 4 stance. The boot's reel-out (1) lets its cable out after toe-off and applies no assist, so it is
  reported as swing. Alongside it: the gait phase, the stride estimate the phase is divided by (the mean of the last
  two strides), heel strikes and torque. `tools/rollout_controllers.py` records all of them on every physics substep.
* **Heel strikes from foot contact force** (the boot uses a shank gyro; its detector is ported too, below). A
  contact counts once the foot has been loaded for `min_contact_time` (50 ms), and is dated back to its start, so
  a toe scuffing the ground in swing is not a heel strike; it ends once the foot has been unloaded for
  `min_unload_time` (50 ms), so a bounce after impact does not split the stride.
* **In the physics loop at 150 Hz.** myosuite writes `data.ctrl` once per 30 Hz control step and runs all 40 physics
  substeps in one call, so a stock env can only act once per step. The device env (`env_id`
  `myoAssistLegImitationExoDevice-v0`, `rl_train/envs/device_control.py`) does what the stock step does, split up:
  the action is processed once exactly as stock, and then on every substep the controller sets the exo's ctrl before
  the sim advances by that substep. The controller ticks every 8th substep (`exo_controller_params.controller_rate_hz`
  150; the boot's loop runs at 175) and holds its torque in between, as the boot's motor holds its last command.
  `device_controller` `"zero"` runs the same loop with the exo off.

## When the assistance starts, and how to change it

Two delays decide when 4PTS first applies torque. Both are fields of `exo_controller_params`: change them in the
config JSON, or override them on the command line as `--config.env_params.exo_controller_params.<field> <value>`.

1. **Once per episode: the warm-up.** No torque until the gait looks steady: `num_strides_required` strides in a row
   (default 2, the boot's `NUM_STRIDES_REQUIRED`), each between `min_stride_duration` and `max_stride_duration`
   (0.6–2.0 s). A stride runs from one heel strike to the next, and the first strike of an episode only starts the
   first stride, so by default assistance starts at the **third** heel strike: about 2.2 s of 1.1 s strides after the
   first one. Training spends that time unassisted in every episode, which `exo_phase_valid` measures (above).

   `num_strides_required 1` starts it at the second strike, about a stride sooner. It also needs
   `num_strides_to_average 1`, because the phase can only be averaged over strides that were checked (the controller
   refuses the combination otherwise):

   ```bash
   python rl_train/run_train.py --config_file_path rl_train/train/train_configs/exoboot_spline/imitation_22_DephyExoBoot_L1_exoboot_spline.json \
     --config.env_params.exo_controller_params.num_strides_required 1 \
     --config.env_params.exo_controller_params.num_strides_to_average 1
   ```

   Then the phase rests on one stride instead of the mean of two, so it follows a stride-to-stride change sooner and
   carries that stride's noise. It cannot start sooner than the second strike: the phase needs one whole stride to
   divide by.

2. **Every stride: the onset.** No torque for `reel_in_time` (0.157 s) after each heel strike, and then the spline,
   which with no bias is zero until `rise_fraction` (0.278) of the stride. With the defaults the rise is what starts
   the torque, 0.278 × 1.1 s ≈ 0.31 s after the strike, so reel-in changes nothing; it matters only with a bias, or
   with a `reel_in_time` past the rise. To move the onset, change `rise_fraction` (it must stay below `peak_fraction`)
   or, with a bias, `reel_in_time`.

## Validated against the boot

`tools/replay_exoboot_log.py` replays a session log through the port, one layer at a time, using the
boot's own logged inputs:

```bash
python tools/replay_exoboot_log.py "<session dir>/<date>_<time>_<subject>_<trial>_" --start 35 --end 240
```

On one participant's four-point-spline session (the validation log), over the steady 1.25 m/s window (35–240 s, 177–178 strides per leg) and the speed-change
window (245–485 s, 206 strides per leg):

| layer | input | result |
|---|---|---|
| spline | logged gait phase | equals the boot's `commanded_torque` exactly (0.0 N·m) |
| phase estimator | logged heel strikes | validity agrees on 100% of rows; phase within 1×10⁻³ (p99) |
| toe-off | logged phase | stance ends on the boot's own row in 100% of strides |
| whole controller | logged heel strikes, each leg's measured reel-in | impulse within 0.2% of the boot's command |
| gyro detector | logged `gyro_z` | 767 of 771 strikes on the boot's own row; the other 4 one or two rows late; none missed or extra |
| 150 Hz in-loop | the device env's tick schedule, with the logged strikes | lag +3.4 to +4.7 ms ± 2.3–2.7 ms; peak and impulse within 0.2% |
| | the same, with the gyro port reading `gyro_z` at each tick | lag +5.5 to +7.0 ms ± 3.1–3.5 ms; peak and impulse within 0.2% |

The replay uses the log's own parameters, including the boot's 3 N·m bias. The phase differences beyond 1×10⁻³ (a
handful of rows) are the boot's own logging: its loop occasionally stalls between stamping `loop_time` and running the
estimator, so those rows log a phase from a later instant. Each of the four late gyro strikes follows such a gap, 8–31
ms between rows where 5.7 ms is usual. A gap is either a stall, in which the boot's `DelayTimer` (it checks
`time.perf_counter()`) runs ahead of the `loop_time` the replay goes by, or loop iterations the boot ran but did not
log, because it only writes a row when the actpack has new data. This log cannot tell the two apart, and feeding the
filter re-sent samples at every gap fixes some of the four and moves others.

In the loop, the lag with the logged strikes is about half a tick (3.3 ms at 150 Hz): each strike is stamped on one of
the boot's rows and seen on the next tick of a separate 150 Hz clock. The gyro port lags 2–3 ms more because it
detects on those ticks instead of on the boot's rows. The env does not use the gyro (above), so the first in-loop row is
the one that applies to training.

Re-run the replay for each boot pair or session before relying on `reel_in_time`. On the validation log it was 150 ms on
the left and 164 ms on the right at 1.25 m/s, and 17–206 ms once the speed was changing.

Checked in the env (`tools/tests/test_device_control.py`): with `"zero"`, the device env reproduces the stock
exo-off env bit for bit — qpos, qvel, act, observation, reward and done on every one of 120 steps (4800 substeps) —
and the same comparison detects 1×10⁻⁶ N·m from the first step. The exo-off baseline is therefore also the in-loop
baseline. Cost (`tools/bench_device_step.py`, one env, 5 × 1000 steps): stock 2.30 ms per step, in-loop with zero
torque 2.62 ms (1.14×). A 4PTS tick in stance costs ~12 µs, nearly all of it scipy's PCHIP, so walking adds at most
~60 µs per step at 5 ticks per step: about 1.17× in all. (Measured with the controller at 175 Hz; 150 Hz only means
fewer ticks.) Envs run in separate processes, so this ratio is the training-throughput ratio.

The boot's own gyro detector is ported (`GyroHeelStrikeDetector`) and checked against the logs (layers 5 and 6 of the
replay), but the env uses foot contact force for heel strikes.

## Tested on a trained policy

`tools/rollout_controllers.py` runs a trained policy in the device env three times from the same start points in the
reference motion: with the exo off, with 4PTS sensing but applying no torque, and with 4PTS. It records every physics
substep and checks the controller against the policy's own foot contacts, and against the same controller run on
every 1200 Hz substep:

```bash
python tools/rollout_controllers.py <train_session_...>/trained_models/<model>.zip
```

It writes `report.md`, `torque_vs_phase.png` and `episodes.npz` to `rl_train/results/rollouts/<time>/`. On the
MyoAssist tutorial policy (`train_session_20250728-161129_tutorial_partial_obs`, `model_19939328.zip`: 19.9 M steps
on `Tutorial_L1`, from the repo's history at `e677989`), which walks on `DephyExoBoot_L1` with the exo off at
1.11 m/s, strides of 1.11 s, for 31–33 s (the 10 longest of 108 start points; about 290 strides per leg):

| check | result |
|---|---|
| strikes | every stance a strike (299 of 299 right, 296 of 296 left), dated +2.8 to +3.0 ± 1.9 ms after the contact began; no other strike: the right foot's 22 toe scuffs in swing are ignored; no stride split |
| phase | valid from the third strike, then 100% of the time |
| no torque | the gait is the exo-off gait, bit for bit |
| applied torque | equals the command (to 2×10⁻¹⁵ N·m) |
| hold | the torque changes only on the 150 Hz ticks |
| timing | per stride, +5.4 to +5.6 ± 2.5 ms behind the controller at 1200 Hz (half a tick to see each strike, half a tick of hold); peak within 0.1%, impulse within 0.5% |

On the true stride the delivered peak lands at phase 0.545–0.557 against the spline's 0.543, ± 0.04–0.07: the boot's
estimator predicts phase from the last two strides, and assisted strides vary in length.

To see it, `tools/render_controller_video.py` renders one of those episodes: the model, each boot tinted red by its
torque, and underneath each ankle's torque, the controller's gait phase estimate, its control state and foot force,
with per leg the two strides the phase is averaged over, their mean, and the current stride's progress against it.
Pick a start index from the rollout report:

```bash
python tools/render_controller_video.py <train_session_...>/trained_models/<model>.zip --start 1280 --seconds 20
```

With 4PTS's 25 N·m, which the policy was never trained with, it falls after 6–20 s (and drifts off the reference
motion, the imitation env's 0.6 rad termination, after 5–14 s), at 1.08–1.20 m/s, with peak plantarflexion of 23–25°
against 18–21° unassisted. Unassisted it walks the full 33 s from all 10 start points, with or without that
termination. A policy trained with 4PTS should do better, but that has not been tested.

The tool runs in evaluate mode at the config's target speed, as the repo's own evaluation does, and starts every
joint from the keyframe: `MyoAssistLegImitation.reset` poses only the reference's joints and leaves the toes, knee
translations and muscle via points where the previous episode ended, which also affects training.

This branch also fixes the target speed in training. From `4a4cbe3` (2025-08-05) until then,
`MyoAssistLegBase._change_mode_and_target_velocity_randomly` passed `set_target_velocity_mode_manually` its arguments
out of order, so after the first reset a training episode's target speed lay anywhere between two random numbers in
[0, 2π] m/s, whatever the config said. Policies trained on earlier code were trained against random target speeds.

## Known limits

* **The contact thresholds are set on one policy's gait.** On the MyoAssist tutorial policy (above) every stance
  peaks at 717–3000 N and passes `grf_on_newtons` (100 N) as it starts, and toe scuffs in swing last 2–30 ms at
  280–700 N, which is why scuffs are told apart by duration (`min_contact_time`), not force. A policy that walks
  differently may need these checked again: `tools/rollout_controllers.py` reports every contact the controller
  counted or missed.
* **The simulated exo is an ideal torque source.** It delivers exactly the command. By the boot's own
  current-based estimate, the boot delivered 3% (steady) to 5–6% (speed changes) more impulse than it commanded:
  ~0.8 N·m of cable tension during reel-in, and a ~15 N·m transient lasting ~17 ms at the start of reel-out.
  In stance it tracks closely: zero lag, peak 25.1 vs 25.0 N·m. That estimate is derived from motor current,
  so it shows the current loop tracking, not torque at the ankle.
* **Reel-in varies stride to stride (±8–9 ms steady, ±26–37 ms during speed changes).** A fixed `reel_in_time`
  leaves 0.6% (steady) to 2% (speed changes) of rows off by more than 1 N·m. That costs nothing in impulse.
