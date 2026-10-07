"""Single-env step cost of the in-loop device-controller seam, against the stock step.

The three-way comparison Hyoungseo asked for, on the same seeded actions:

* **stock**: the exo env under the exo-off config, myosuite's own step;
* **device, zero torque**: the device env with a controller that applies nothing -- the cost of the split-up step alone;
* **device, ExoBoot 4PTS 150 Hz**: the device env with the ExoBoot four-point spline;

Training runs its envs in separate processes (SubprocVecEnv), so the overhead is per env and the single-env ratio
is the training-throughput ratio. Only ``env.step`` is timed; resets are not. Rounds interleave the cases so that
drift in machine load hits all three alike.

On a CPU with performance and efficiency cores (or on battery power) the scheduler moves the process between cores
of different speeds, and the ratios swing by 2x from run to run. ``--cpu`` pins the process to one logical CPU (a
performance core), which makes them repeatable.

    python tools/bench_device_step.py [--steps 1000] [--rounds 5] [--cpu N]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "rl_train/train/train_configs/exoboot_spline"
EXO_OFF = CONFIG_DIR / "imitation_22_DephyExoBoot_L1_exo_off.json"
SPLINE = CONFIG_DIR / "imitation_22_DephyExoBoot_L1_exoboot_spline.json"
DEVICE_ENV_ID = "myoAssistLegImitationExoDevice-v0"

CASES = {
    "stock": (EXO_OFF, {}),
    "device, zero torque": (EXO_OFF, {"env_id": DEVICE_ENV_ID, "device_controller": "zero"}),
    "device, ExoBoot 4PTS 150 Hz": (SPLINE, {}),
}


def build(path: pathlib.Path, overrides: dict):
    from rl_train.envs.environment_handler import EnvironmentHandler
    from rl_train.train.train_configs.config_imiatation_exo import (
        ExoImitationTrainSessionConfig,
    )
    from rl_train.utils.data_types import DictionableDataclass

    config = DictionableDataclass.create(ExoImitationTrainSessionConfig, json.loads(path.read_text()))
    config.env_params.num_envs = 1
    for key, value in overrides.items():
        setattr(config.env_params, key, value)
    return EnvironmentHandler.create_environment(config, is_rendering_on=False)


def time_steps(env, actions: np.ndarray, seed: int) -> list[float]:
    """Seconds per ``env.step`` over ``actions``, resetting (untimed) whenever an episode ends."""
    random.seed(seed)
    np.random.seed(seed)
    env.reset()
    seconds = []
    for a in actions:
        start = time.perf_counter()
        _, _, terminated, truncated, _ = env.step(a)
        seconds.append(time.perf_counter() - start)
        if terminated or truncated:
            env.reset()
    return seconds


def main(argv=None) -> dict[str, float]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=1000, help="timed steps per case per round")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--cpu", type=int, help="pin the process to this logical CPU (needs psutil)")
    args = parser.parse_args(argv)
    if args.cpu is not None:
        import psutil

        psutil.Process().cpu_affinity([args.cpu])

    envs = {name: build(path, overrides) for name, (path, overrides) in CASES.items()}
    try:
        any_env = next(iter(envs.values()))
        exo_ids = [any_env.sim.model.actuator(n).id for n in ("Exo_R", "Exo_L")]
        nu = any_env.sim.model.nu
        per_case = {name: [] for name in envs}
        for r in range(args.rounds + 1):  # round 0 warms up and is discarded
            actions = np.random.default_rng(r).uniform(-1.0, 1.0, size=(args.steps, nu))
            actions[:, exo_ids] = 1.0
            for name, env in envs.items():
                seconds = time_steps(env, actions, seed=r)
                if r:
                    per_case[name].extend(seconds)
    finally:
        for env in envs.values():
            env.close()

    stock = statistics.fmean(per_case["stock"])
    pinned = "" if args.cpu is None else f", pinned to CPU {args.cpu}"
    print(f"{args.rounds} rounds x {args.steps} steps per case, single env, 40 physics substeps per step{pinned}\n")
    print(f"{'case':32s} {'mean us/step':>13s} {'median':>9s} {'steps/s':>9s} {'vs stock':>9s}")
    means = {}
    for name, seconds in per_case.items():
        mean = statistics.fmean(seconds)
        means[name] = mean
        print(f"{name:32s} {mean * 1e6:13.0f} {statistics.median(seconds) * 1e6:9.0f} {1 / mean:9.0f} {mean / stock:8.2f}x")
    return means


if __name__ == "__main__":
    main()
