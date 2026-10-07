"""Build each leg's ExoBoot four-point spline controller from config and a composed model."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from myoassist_utils.exo_ctrl.base import LegExoController
from myoassist_utils.exo_ctrl.fourpoint_spline import (
    ExoBootFourPointSplineController,
    FourPointSpline,
)
from myoassist_utils.exo_ctrl.phase import (
    StrideAverageGaitPhaseEstimator,
    VgrfHeelStrikeDetector,
)
from myoassist_utils.exo_ctrl.torque_adapter import ankle_torque_actuator

HEEL_STRIKE_SOURCES = ("vgrf",)


@dataclass
class LegExo:
    """One leg's exo actuator and the controller that drives it."""

    side: str  # "r" or "l", as in MyoAssistLegBase._get_foot_force
    actuator_id: int
    controller: LegExoController


def build_leg_exos(params, model: mujoco.MjModel) -> list[LegExo]:
    """One ``LegExo`` per side, right then left.

    ``params`` is duck-typed -- anything with the fields of
    ``ExoImitationTrainSessionConfig.EnvParams.ExoControllerParams`` -- so that this module does not import rl_train.
    """
    if params.heel_strike_source not in HEEL_STRIKE_SOURCES:
        raise ValueError(f"unknown heel_strike_source {params.heel_strike_source!r}; expected one of {HEEL_STRIKE_SOURCES}")

    legs = []
    for side, actuator_name in (
        ("r", params.right_actuator),
        ("l", params.left_actuator),
    ):
        controller = ExoBootFourPointSplineController(
            spline=FourPointSpline(
                rise_fraction=params.rise_fraction,
                peak_fraction=params.peak_fraction,
                fall_fraction=params.fall_fraction,
                peak_torque=params.peak_torque,
                bias_torque=params.bias_torque,
                peak_hold_time=params.peak_hold_time,
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
            toe_off_fraction=params.toe_off_fraction,
            reel_in_time=params.reel_in_time,
        )
        legs.append(LegExo(side=side, actuator_id=ankle_torque_actuator(model, actuator_name, side), controller=controller))
    return legs
