"""Scripted exoskeleton controllers that run inside the RL environment in place of learned exo torque.

Ported from the NeuMove ExoBoot hardware controller so that the same parameters drive the simulated
and the physical Dephy boot. See ``rl_train/train/train_configs/exoboot_spline/README.md``.
"""

from myoassist_utils.exo_ctrl.base import HeelStrikeDetector, LegExoController
from myoassist_utils.exo_ctrl.boot_filters import Butterworth, DelayTimer
from myoassist_utils.exo_ctrl.device import (
    DEVICE_CONTROLLERS,
    FixedRateLegExos,
    FootForce,
    ZeroTorqueDevice,
    build_device_controller,
)
from myoassist_utils.exo_ctrl.factory import LegExo, build_leg_exos
from myoassist_utils.exo_ctrl.fourpoint_spline import ExoBootFourPointSplineController, FourPointSpline
from myoassist_utils.exo_ctrl.phase import GyroHeelStrikeDetector, StrideAverageGaitPhaseEstimator, VgrfHeelStrikeDetector
from myoassist_utils.exo_ctrl.schedule import TickSchedule
from myoassist_utils.exo_ctrl.torque_adapter import ankle_torque_actuator, torque_actuator_params

__all__ = [
    "DEVICE_CONTROLLERS",
    "Butterworth",
    "DelayTimer",
    "ExoBootFourPointSplineController",
    "FixedRateLegExos",
    "FootForce",
    "FourPointSpline",
    "GyroHeelStrikeDetector",
    "HeelStrikeDetector",
    "LegExo",
    "LegExoController",
    "StrideAverageGaitPhaseEstimator",
    "TickSchedule",
    "VgrfHeelStrikeDetector",
    "ZeroTorqueDevice",
    "ankle_torque_actuator",
    "build_device_controller",
    "build_leg_exos",
    "torque_actuator_params",
]
