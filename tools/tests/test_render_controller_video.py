"""tools/render_controller_video.py: what its stride-average panel shows, which geoms it tints, and that its strip of
plots draws with each controller's panel.

Rendering the model needs OpenGL and is not run here; the plots under it do not.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

PARAMS = types.SimpleNamespace(
    peak_torque=25.0,
    rise_fraction=0.278,
    peak_fraction=0.543,
    toe_off_fraction=0.6,
    reel_in_time=0.157,
    grf_on_newtons=100.0,
    num_strides_required=2,
    min_stride_duration=0.6,
    max_stride_duration=2.0,
)


def _rows(fields, seconds=5.0, rate=1200.0):
    """A record of a 1.1 s gait: strikes on the tick at each stride's start, phase from the third, the boot's states."""
    t = np.arange(0.0, seconds, 1 / rate)
    since = t % 1.1
    rows = np.full((len(t), len(fields)), np.nan)
    col = {name: i for i, name in enumerate(fields)}
    rows[:, col["t"]] = t
    rows[:, col["tick"]] = (np.arange(len(t)) % 8 == 0).astype(float)
    for side in ("r", "l"):
        valid = t >= 2.2
        rows[:, col[f"strike_{side}"]] = (since < 1 / rate).astype(float)
        rows[:, col[f"strike_time_{side}"]] = np.where(since < 1 / rate, t, np.nan)
        rows[:, col[f"phase_{side}"]] = np.where(valid, since / 1.1, -1.0)
        rows[:, col[f"valid_{side}"]] = valid.astype(float)
        rows[:, col[f"stride_estimate_{side}"]] = np.where(valid, 1.1, -1.0)
        rows[:, col[f"control_state_{side}"]] = np.where(valid & (since > 0.157) & (since < 0.66), 4.0, 2.0)
        rows[:, col[f"cmd_{side}"]] = np.where(rows[:, col[f"control_state_{side}"]] == 4.0, 10.0, 0.0)
        rows[:, col[f"grf_{side}"]] = np.where(since < 0.66, 800.0, 0.0)
    return rows


@pytest.mark.parametrize("controller", ["exoboot_spline", "zero"])
def test_the_strip_draws_with_each_controllers_panel(controller):
    from tools.render_controller_video import PANELS, WIDTH, NoControllerPanel, Recorder, Strip

    panel = PANELS.get(controller, NoControllerPanel)
    strip = Strip(PARAMS, controller, panel, Recorder.FIELDS)
    rows = _rows(Recorder.FIELDS)
    if controller == "zero":
        rows[:, 8:] = np.nan  # no controller diagnostics with the exo off
    frame = strip.draw(rows, now=float(rows[-1, 0]), start=0)
    assert frame.shape[1] == WIDTH and frame.shape[2] == 3 and frame.std() > 0


def test_the_panel_shows_the_strides_the_estimate_averages():
    from tools.render_controller_video import stride_panel

    panel = stride_panel(np.array([0.5, 1.6, 2.7, 3.9]), now=4.3, estimate=1.15)
    np.testing.assert_allclose(
        [panel["stride before last"], panel["last stride"], panel["estimate (mean)"], panel["this stride"]],
        [1.1, 1.2, 1.15, 0.4],
        atol=1e-12,
    )


def test_before_two_strides_there_is_nothing_to_average():
    from tools.render_controller_video import stride_panel

    assert stride_panel(np.array([]), now=1.0, estimate=-1.0) == dict.fromkeys(
        ["stride before last", "last stride", "estimate (mean)", "this stride"], 0.0
    )
    one = stride_panel(np.array([0.5, 1.6]), now=2.0, estimate=-1.0)
    assert one["stride before last"] == 0.0 and one["estimate (mean)"] == 0.0
    np.testing.assert_allclose([one["last stride"], one["this stride"]], [1.1, 0.4])


def test_each_legs_boot_geoms_are_tinted_and_nothing_else():
    import mujoco

    from myoassist_utils.compose import compose_env_model
    from tools.render_controller_video import boot_geoms

    model = mujoco.MjModel.from_xml_string(compose_env_model("myolegs22", "DephyExoBoot_L1", terrain=None))
    names = {side: {model.geom(i).name for i in boot_geoms(model, "DephyExoBoot_L1", side)} for side in ("r", "l")}
    assert {"DephyExoBoot_L1_exo_1_r_geom", "DephyExoBoot_L1_tb_shank_r_geom"} <= names["r"]
    assert {"DephyExoBoot_L1_exo_1_l_geom", "DephyExoBoot_L1_tb_shank_l_geom"} <= names["l"]
    assert not names["r"] & names["l"]
    assert all(name.endswith("_r_geom") for name in names["r"]), "only the right leg's parts"
