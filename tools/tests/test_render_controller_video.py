"""tools/render_controller_video.py: what its stride-average panel shows, and which geoms it tints.

Rendering itself needs OpenGL and is not run here.
"""

from __future__ import annotations

import numpy as np


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
