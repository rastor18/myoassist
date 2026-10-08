"""What the ExoBoot's controllers share around their stance controller: its state machine, and its ankle reading.

The state machine is ``StanceSwingReeloutReelinStateMachine.step``; the encoder is the ankle half of ``Exo.read_data``.
Both are used by more than one ported controller, so their behaviour is pinned down here, apart from any of them.
"""

from __future__ import annotations

import math
import types

import mujoco
import pytest

from myoassist_utils.compose import compose_env_model
from myoassist_utils.exo_ctrl import AnkleEncoder, BootStateMachine
from myoassist_utils.exo_ctrl.boot_state import ENCODER_CLICK_DEG, REEL_IN, REEL_OUT, STANCE, SWING

# --- state machine ---------------------------------------------------------------------------------------------------


def _machine():
    return BootStateMachine(reel_in_time=0.15, reel_out_time=0.17)


def test_the_state_cycle():
    m = _machine()

    def step(t, **events):
        return m.step(t, **{"did_heel_strike": False, "did_toe_off": False, "gait_phase": 0.1, "swing_only": False, **events})

    assert step(0.0) == REEL_OUT, "the boot starts by reeling out"
    assert step(0.17) == REEL_OUT, "reel-out lasts strictly longer than its time"
    assert step(0.171) == SWING
    assert step(0.5, gait_phase=None, did_heel_strike=True) == SWING, "no heel strike without a gait phase"
    assert step(0.6, did_heel_strike=True) == REEL_IN
    assert step(0.75) == REEL_IN
    assert step(0.751) == STANCE
    assert step(1.0) == STANCE
    assert step(1.2, did_toe_off=True) == REEL_OUT, "toe-off ends stance on the same tick"
    assert step(1.371) == SWING


def test_a_lost_phase_ends_stance_and_swing_only_overrides_everything():
    m = _machine()
    base = dict(did_heel_strike=False, did_toe_off=False, gait_phase=0.1, swing_only=False)
    m.step(0.0, **base)
    m.step(0.2, **base)
    m.step(0.3, **{**base, "did_heel_strike": True})
    assert m.step(0.5, **base) == STANCE
    assert m.step(0.6, **{**base, "gait_phase": None}) == REEL_OUT
    assert m.step(0.65, **{**base, "swing_only": True}) == SWING
    assert m.step(0.7, **{**base, "swing_only": True, "did_heel_strike": True}) == SWING


def test_reset_starts_the_machine_over():
    m = _machine()
    base = dict(did_heel_strike=False, did_toe_off=False, gait_phase=0.1, swing_only=False)
    m.step(0.0, **base)
    m.step(0.2, **base)
    m.reset()
    assert m.state is None
    assert m.step(5.0, **base) == REEL_OUT, "after a reset the boot starts by reeling out again"


@pytest.mark.parametrize("field", ["reel_in_time", "reel_out_time"])
def test_a_negative_duration_is_refused(field):
    with pytest.raises(ValueError, match=field):
        BootStateMachine(**{"reel_in_time": 0.15, "reel_out_time": 0.17, field: -0.01})


# --- ankle encoder ---------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(compose_env_model("myolegs22", "DephyExoBoot_L1"))


def _standing(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    return data, types.SimpleNamespace(data=data)


def test_the_angle_is_plantarflexion_positive_from_the_standing_angle(model):
    data, sim = _standing(model)
    encoder = AnkleEncoder(model, "l", standing_angle_deg=-8.0, quantize=False)
    assert encoder.read(sim)[0] == pytest.approx(-8.0, abs=1e-9), "the standing keyframe reads the standing angle"
    data.qpos[model.jnt_qposadr[model.joint("ankle_angle_l").id]] += 0.1  # dorsiflex by 0.1 rad
    assert encoder.read(sim)[0] == pytest.approx(-8.0 - math.degrees(0.1), abs=1e-9)


def test_each_side_reads_its_own_ankle(model):
    data, sim = _standing(model)
    right = AnkleEncoder(model, "r", standing_angle_deg=0.0, quantize=False)
    left = AnkleEncoder(model, "l", standing_angle_deg=0.0, quantize=False)
    data.qpos[model.jnt_qposadr[model.joint("ankle_angle_r").id]] -= 0.2
    assert right.read(sim)[0] == pytest.approx(math.degrees(0.2), abs=1e-9)
    assert left.read(sim)[0] == pytest.approx(0.0, abs=1e-9)


def test_quantization_is_whole_encoder_clicks_before_the_offset(model):
    data, sim = _standing(model)
    encoder = AnkleEncoder(model, "r", standing_angle_deg=-1.87)
    data.qpos[model.jnt_qposadr[model.joint("ankle_angle_r").id]] -= 0.123
    clicks = (encoder.read(sim)[0] - encoder.offset_deg) / ENCODER_CLICK_DEG
    assert clicks == pytest.approx(round(clicks), abs=1e-6)
    exact = AnkleEncoder(model, "r", standing_angle_deg=-1.87, quantize=False).read(sim)[0]
    assert abs(encoder.read(sim)[0] - exact) <= ENCODER_CLICK_DEG / 2 + 1e-12


def test_velocity_is_zero_on_the_first_read_then_filtered(model):
    data, sim = _standing(model)
    encoder = AnkleEncoder(model, "r", standing_angle_deg=0.0, quantize=False)
    adr = model.jnt_qposadr[model.joint("ankle_angle_r").id]
    assert encoder.read(sim)[1] == 0.0
    data.time += 1 / 175
    data.qpos[adr] -= math.radians(1.0)  # 1 deg of plantarflexion in one tick
    # The filter starts from its first input, so the first filtered value is the raw difference quotient.
    assert encoder.read(sim)[1] == pytest.approx(175.0, rel=1e-9)
    data.time += 1 / 175
    assert 0.0 < encoder.read(sim)[1] < 175.0, "then the 10 Hz low-pass smooths a stop"
    encoder.reset()
    assert encoder.read(sim)[1] == 0.0


def test_a_read_at_the_same_time_keeps_the_velocity(model):
    """Two reads within one sim instant leave the velocity as it was: a difference over no time is undefined."""
    data, sim = _standing(model)
    encoder = AnkleEncoder(model, "r", standing_angle_deg=0.0, quantize=False)
    encoder.read(sim)
    data.time += 1 / 175
    data.qpos[model.jnt_qposadr[model.joint("ankle_angle_r").id]] -= math.radians(1.0)
    first = encoder.read(sim)[1]
    assert encoder.read(sim)[1] == first
