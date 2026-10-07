"""The ankle exo actuators a scripted controller may drive, and the sign of plantarflexion on them."""

from __future__ import annotations

import mujoco

# Plantarflexion is negative joint torque on the MyoAssist ankles, whose ankle_angle is dorsiflexion-positive.
# Measured on myolegs22 x DephyExoBoot_L1: ctrl = -1 on Exo_R gave qfrc_actuator = -99.95 N*m and moved the
# ankle from -0.014 to -0.572 rad.
PLANTARFLEXION_SIGN = -1.0
# The sign above is only established for these joints; the MyoAssist leg models name both ankles this way.
ANKLE_JOINT_PREFIX = "ankle_angle"


def torque_actuator_params(model: mujoco.MjModel, actuator_id: int) -> tuple[float, float, float, float]:
    """``(gain, gear, ctrl_low, ctrl_high)`` of an actuator whose joint torque is ``gain * gear * ctrl``.

    That holds only for a fixed-gain, bias-free, dynamics-free actuator on a joint. A filtered actuator
    (UTAnkleExo_L2) or a tendon drive in newtons (STRIDE_L2) would get a wrong torque without any error, so
    they are rejected, as is an actuator without a ctrlrange, whose action normalization is undefined.
    """
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    checks = {
        "trntype": (model.actuator_trntype[actuator_id], mujoco.mjtTrn.mjTRN_JOINT),
        "gaintype": (model.actuator_gaintype[actuator_id], mujoco.mjtGain.mjGAIN_FIXED),
        "biastype": (model.actuator_biastype[actuator_id], mujoco.mjtBias.mjBIAS_NONE),
        "dyntype": (model.actuator_dyntype[actuator_id], mujoco.mjtDyn.mjDYN_NONE),
    }
    wrong = {field: int(actual) for field, (actual, expected) in checks.items() if actual != expected}
    if wrong:
        raise ValueError(
            f"actuator {name!r} is not a fixed-gain, bias-free, dynamics-free joint actuator "
            f"({wrong}), so torque cannot be mapped to ctrl as gain * gear * ctrl"
        )
    if not model.actuator_ctrllimited[actuator_id]:
        raise ValueError(f"actuator {name!r} is not ctrllimited, so its action normalization is undefined")
    ctrl_low, ctrl_high = model.actuator_ctrlrange[actuator_id]
    return (
        float(model.actuator_gainprm[actuator_id, 0]),
        float(model.actuator_gear[actuator_id, 0]),
        float(ctrl_low),
        float(ctrl_high),
    )


def ankle_torque_actuator(model: mujoco.MjModel, actuator_name: str, side: str) -> int:
    """The id of ``actuator_name``, refusing an actuator a scripted ankle controller would drive wrongly.

    On top of ``torque_actuator_params``'s checks, the actuator must drive this side's ankle, ``ankle_angle_{side}``:
    a hip exo such as Hippo_L1 passes every other check and ``PLANTARFLEXION_SIGN`` means nothing there, and a left
    actuator named for the right leg would give the wrong leg a plausible torque.
    """
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if actuator_id < 0:
        raise KeyError(f"no actuator named {actuator_name!r} in the model")
    torque_actuator_params(model, actuator_id)
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[actuator_id, 0]))
    expected = f"{ANKLE_JOINT_PREFIX}_{side}"
    if joint_name != expected:
        raise ValueError(
            f"actuator {actuator_name!r} drives joint {joint_name!r}, not the {side} ankle ({expected}); a "
            "plantarflexion torque for that leg has no meaning there"
        )
    return int(actuator_id)
