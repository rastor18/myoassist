"""run_train.py must accept --config.* overrides for every field of the config class the run actually uses.

It used to build the flags from TrainSessionConfigBase and only then resolve the real class from env_id, so
any field that only a subclass defines was rejected as an unrecognized argument -- out_of_trajectory_threshold
(imitation), and the scripted exo controller params (exo imitation). The documented examples all used base
fields such as num_envs, so nothing caught it.
"""

from __future__ import annotations

import pytest

from tools.tests.conftest import REPO_ROOT

SPLINE = str(REPO_ROOT / "rl_train/train/train_configs/exoboot_spline/imitation_22_DephyExoBoot_L1_exoboot_spline.json")


@pytest.mark.parametrize(
    "flag, value, read",
    [
        (
            "--config.env_params.num_envs",
            "3",
            lambda c: c.env_params.num_envs,
        ),  # base class: always worked
        # ImitationTrainSessionConfig only:
        (
            "--config.env_params.out_of_trajectory_threshold",
            "0.55",
            lambda c: c.env_params.out_of_trajectory_threshold,
        ),
        # ExoImitationTrainSessionConfig only:
        (
            "--config.env_params.exo_controller_params.bias_torque",
            "3",
            lambda c: c.env_params.exo_controller_params.bias_torque,
        ),
        (
            "--config.env_params.exo_controller_params.toe_off_fraction",
            "0.62",
            lambda c: c.env_params.exo_controller_params.toe_off_fraction,
        ),
        (
            "--config.env_params.reward_keys_and_weights.exo_phase_valid",
            "0.5",
            lambda c: c.env_params.reward_keys_and_weights.exo_phase_valid,
        ),
    ],
)
def test_subclass_fields_are_overridable(flag, value, read):
    from rl_train.run_train import parse_args_and_config

    _, config = parse_args_and_config(["--config_file_path", SPLINE, flag, value])
    assert read(config) == pytest.approx(float(value)), f"{flag} {value} did not reach the config"


def test_boolean_subclass_field_can_be_negated():
    """A bool only the imitation subclass defines, true in the config: --no-... turns it off."""
    from rl_train.run_train import parse_args_and_config

    _, config = parse_args_and_config(
        [
            "--config_file_path",
            SPLINE,
            "--no-config.env_params.flag_random_ref_index",
        ]
    )
    assert config.env_params.flag_random_ref_index is False
