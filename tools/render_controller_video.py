"""Render a video of a policy walking with an in-loop exo controller, and of what the controller is doing underneath.

The model on top, each boot tinted red by its exo torque. Under it, over a rolling 4 s window: each ankle's exo
torque, the controller's gait phase estimate, its control state (the boot's: swing, reel-in, stance) and foot force.
Beside them, per leg, the stride average the phase comes from: the two strides it averages, their mean, and this
stride's elapsed time against it, with where reel-in, rise, peak and toe-off fall.

The episode starts at one index of the reference motion, as ``tools/rollout_controllers.py`` runs them (its report
lists the start indices that walk). As in the repo's own evaluation videos, only a fall or the episode limit ends it,
not drifting off the reference motion.

    python tools/render_controller_video.py <policy.zip> [--config <4PTS config>] [--start 0] [--case 4PTS]
        [--seconds 20] [--out <file.mp4>]

Rendering is offscreen OpenGL; on a headless Linux machine set MUJOCO_GL=egl (or osmesa).
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

import numpy as np

# Its sibling tool, importable whether this runs as a script or is imported as tools.render_controller_video.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rollout_controllers import CASES, DEFAULT_CONFIG, Recorder, load_policy, make_env, reset_at  # noqa: E402

SIDES = ("r", "l")
RIGHT, LEFT, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#52514e", "#898781", "#e6e5e0"
COLOR = {"r": RIGHT, "l": LEFT}
STATES = {2: ("swing", "#d9d8d2"), 3: ("reel-in", "#eda100"), 4: ("stance (spline)", "#1baf7a")}
COL = {name: i for i, name in enumerate(Recorder.FIELDS)}
WINDOW = 4.0  # seconds of history shown
WIDTH, HEIGHT = 1280, 560  # the 3D view; the strip under it is as wide


def stride_panel(strikes: np.ndarray, now: float, estimate: float) -> dict[str, float]:
    """The stride-average panel's bars, in seconds: the stride before last and the last stride (what a two-stride
    average is taken over), the controller's estimate, and the time since the last strike. 0 for what does not exist
    yet; the estimate is the controller's own, reported as -1 before it has one."""
    strides = np.diff(strikes)
    return {
        "stride before last": float(strides[-2]) if len(strides) >= 2 else 0.0,
        "last stride": float(strides[-1]) if len(strides) >= 1 else 0.0,
        "estimate (mean)": float(estimate) if estimate > 0 else 0.0,
        "this stride": float(now - strikes[-1]) if len(strikes) else 0.0,
    }


def boot_geoms(model, device_key: str, side: str) -> list[int]:
    """The device's geoms on one leg (named ``<device_key>_..._<side>_geom``, as the composed exos name them)."""
    pattern = re.compile(rf"^{re.escape(device_key)}_.*_{side}_geom$")
    return [i for i in range(model.ngeom) if pattern.match(model.geom(i).name or "")]


class Strip:
    """The plots under the 3D view, redrawn every frame from the ``Recorder``'s rows so far."""

    def __init__(self, params, label: str):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        self.p, self.label = params, label
        self.fig = fig = plt.figure(figsize=(WIDTH / 100, 6.6), dpi=100, layout="constrained")
        grid = fig.add_gridspec(4, 2, width_ratios=[2.45, 1], height_ratios=[1, 1, 0.42, 0.8])
        self.ax_t = ax_t = fig.add_subplot(grid[0, 0])
        ax_p = fig.add_subplot(grid[1, 0], sharex=ax_t)
        ax_s = fig.add_subplot(grid[2, 0], sharex=ax_t)
        ax_f = fig.add_subplot(grid[3, 0], sharex=ax_t)
        self.ax_est = {"r": fig.add_subplot(grid[0:2, 1]), "l": fig.add_subplot(grid[2:4, 1])}
        for ax in (ax_t, ax_p, ax_s, ax_f, *self.ax_est.values()):
            ax.grid(color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

        self.lines = {}
        for side, name in (("r", "right"), ("l", "left")):
            self.lines[f"t{side}"] = ax_t.plot([], [], color=COLOR[side], lw=2, label=name)[0]
            self.lines[f"p{side}"] = ax_p.plot([], [], color=COLOR[side], lw=2)[0]
            self.lines[f"f{side}"] = ax_f.plot([], [], color=COLOR[side], lw=1.2)[0]
        ax_t.set(ylim=(-1, 1.4 * max(params.peak_torque, 10.0)), ylabel="exo torque\n(N·m)")
        ax_t.legend(
            handles=[self.lines["tr"], self.lines["tl"], *[Patch(color=c, label=f"state: {n}") for n, c in STATES.values()]],
            loc="upper left",
            frameon=False,
            fontsize=7.5,
            ncol=5,
        )
        ax_p.set(ylim=(-0.03, 1.08), yticks=[0, 0.5, 1.0], ylabel="gait phase\nestimate")
        self.level_labels = []
        for level, name, va in (
            (params.rise_fraction, "rise", "center"),
            (params.peak_fraction, "peak", "top"),
            (params.toe_off_fraction, "toe-off", "bottom"),
        ):
            ax_p.axhline(level, color=MUTED, lw=0.8, ls="--")
            self.level_labels.append(ax_p.text(0, level, f"{name} {level:g}", fontsize=6.5, color=INK, va=va))
        self.band = ax_s.imshow(
            np.full((2, 2), np.nan),
            aspect="auto",
            cmap=ListedColormap([color for _, color in STATES.values()]),
            vmin=1.5,
            vmax=4.5,
            interpolation="nearest",
        )
        ax_s.set_yticks([0, 1])
        ax_s.set_yticklabels(["right", "left"], fontsize=8)
        ax_s.set_ylabel("control\nstate")
        ax_s.grid(False)
        ax_f.set(ylim=(-50, 2000), ylabel="foot force\n(N)", xlabel="time (s)")
        ax_f.axhline(params.grf_on_newtons, color=MUTED, lw=0.8, ls="--")
        self.cursors = [ax.axvline(0, color=INK, lw=1) for ax in (ax_t, ax_p, ax_f)]
        for ax in (ax_t, ax_p, ax_s):
            plt.setp(ax.get_xticklabels(), visible=False)

        self.rows_y = {"stride before last": 3, "last stride": 2, "estimate (mean)": 1, "this stride": 0}
        self.est = {}
        above = {"reel-in": True, "rise": False, "peak": True, "toe-off": False, "end": True}
        for side, name in (("r", "Right"), ("l", "Left")):
            ax = self.ax_est[side]
            bars = ax.barh(list(self.rows_y.values()), [0, 0, 0, 0], height=0.62, color=[COLOR[side], COLOR[side], INK, MUTED])
            ax.set_yticks(list(self.rows_y.values()))
            ax.set_yticklabels(list(self.rows_y), fontsize=8)
            ax.set_xlim(0, 1.45)
            ax.set_ylim(-1.25, 3.6)
            ax.set_title(f"{name} leg: the stride average", fontsize=9, loc="left")
            ax.tick_params(axis="x", labelsize=7)
            ax.set_xlabel("seconds", fontsize=8)
            marks = {key: ax.plot([], [], color="#eda100" if key == "reel-in" else INK, lw=2)[0] for key in above}
            labels = {
                key: ax.text(0, 0.4 if up else -0.4, "", fontsize=6.5, ha="center", va="bottom" if up else "top")
                for key, up in above.items()
            }
            values = [ax.text(0, y, "", fontsize=7.5, va="center", ha="left", color=INK) for y in self.rows_y.values()]
            note = ax.text(0.0, -1.05, "", fontsize=8, color=INK)
            self.est[side] = (bars, marks, labels, values, note)
        self.title = fig.suptitle("", fontsize=10)

    def _update_estimator(self, side: str, rows: np.ndarray, now: float) -> None:
        p = self.p
        bars, marks, labels, values, note = self.est[side]
        last = rows[-1]
        if not np.isfinite(last[COL[f"control_state_{side}"]]):
            note.set_text("no controller")
            return
        reported = (rows[:, COL["tick"]] == 1) & (rows[:, COL[f"strike_{side}"]] == 1)
        estimate = last[COL[f"stride_estimate_{side}"]]
        widths = stride_panel(rows[reported, COL[f"strike_time_{side}"]], now, estimate)
        state = int(last[COL[f"control_state_{side}"]])
        for bar, width in zip(bars, widths.values()):
            bar.set_width(width)
        for bar, width in zip(bars[:2], list(widths.values())[:2]):
            bar.set_color(COLOR[side] if p.min_stride_duration < width < p.max_stride_duration else "#c3c2b7")
        bars[3].set_color(STATES[state][1])
        for text, width, y in zip(values, widths.values(), self.rows_y.values()):
            text.set_position((width + 0.02, y))
            text.set_text(f"{width:.2f} s" if width > 0 else "")
        if estimate > 0:
            points = {
                "reel-in": p.reel_in_time,
                "rise": p.rise_fraction * estimate,
                "peak": p.peak_fraction * estimate,
                "toe-off": p.toe_off_fraction * estimate,
                "end": estimate,
            }
            for key, x in points.items():
                marks[key].set_data([x, x], [-0.36, 0.36])
                labels[key].set_position((x, labels[key].get_position()[1]))
                labels[key].set_text(key if key != "end" else "")
        if last[COL[f"valid_{side}"]] != 1:
            note.set_text(
                f"no phase yet: needs {p.num_strides_required} whole strides in "
                f"[{p.min_stride_duration:g}, {p.max_stride_duration:g}] s"
            )
        else:
            note.set_text(
                f"phase = {widths['this stride']:.2f} s since strike / {estimate:.2f} s = "
                f"{last[COL[f'phase_{side}']]:.2f}: {STATES[state][0]}"
            )

    def draw(self, rows: np.ndarray, now: float, start: int) -> np.ndarray:
        t = rows[:, COL["t"]]
        shown = t >= now - WINDOW
        for side in SIDES:
            self.lines[f"t{side}"].set_data(t[shown], rows[shown, COL[f"cmd_{side}"]])
            phase = rows[shown, COL[f"phase_{side}"]]
            self.lines[f"p{side}"].set_data(t[shown], np.where(phase >= 0, phase, np.nan))
            self.lines[f"f{side}"].set_data(t[shown], rows[shown, COL[f"grf_{side}"]])
            self._update_estimator(side, rows, now)
        sampled = np.linspace(now - WINDOW, now, 400)
        k = np.clip(np.searchsorted(t, sampled), 0, len(t) - 1)
        states = np.vstack([rows[k, COL["control_state_r"]], rows[k, COL["control_state_l"]]])
        states[:, sampled < t[0]] = np.nan
        self.band.set_data(states)
        self.band.set_extent((now - WINDOW, now, 1.5, -0.5))
        for cursor in self.cursors:
            cursor.set_xdata([now, now])
        self.ax_t.set_xlim(now - WINDOW, now + 0.55)
        for text in self.level_labels:
            text.set_x(now + 0.04)
        self.title.set_text(f"{self.label}, start {start}    t = {now:4.1f} s")
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3][:, :WIDTH]


def main(argv=None):
    import imageio
    import mujoco

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("policy", type=pathlib.Path, help="a trained policy's .zip")
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG, help="a 4PTS device-env config")
    parser.add_argument("--start", type=int, default=0, help="the reference-motion index the episode starts at")
    parser.add_argument("--case", choices=list(CASES), default="4PTS")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)
    out = args.out or pathlib.Path("rl_train/results/rollouts") / f"video_{args.case.replace(' ', '_')}_{args.start}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    policy = load_policy(args.policy)
    env, config = make_env(args.config, CASES[args.case])
    env._out_of_trajectory_threshold = float("inf")  # only a fall or the episode limit ends it
    params = config.env_params.exo_controller_params
    model = env.sim.model
    tinted = {side: boot_geoms(model, config.env_params.device_key, side) for side in SIDES}
    base_rgba = {side: model.geom_rgba[ids].copy() for side, ids in tinted.items()}
    red = np.array([0.85, 0.12, 0.10, 1.0])

    # No shadows or floor reflections, for a clean side view. Rendering only: neither enters the physics.
    model.light_castshadow[:] = 0
    model.mat_reflectance[:] = 0
    renderer = env.sim.renderer
    cam = mujoco.MjvCamera()
    label = args.case if args.case == "exo off" else f"{args.case} ({params.peak_torque:g} N·m peak)"
    strip = Strip(params, label)

    obs, _ = reset_at(env, args.start)
    frames, ending = [], f"{args.seconds:g} s shown"
    for _ in range(int(args.seconds / env.dt)):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        rows = np.asarray(env.device_controller._rows)
        now = float(env.sim.data.time)
        for side in SIDES:
            level = float(np.clip(rows[-1, COL[f"cmd_{side}"]] / max(params.peak_torque, 1e-9), 0.0, 1.0))
            model.geom_rgba[tinted[side]] = base_rgba[side] * (1 - level) + red * level
        pelvis = env.sim.data.body("pelvis").xpos.copy()
        cam.distance, cam.azimuth, cam.elevation, cam.lookat = 3.0, 90.0, -8.0, np.array([pelvis[0], pelvis[1], 0.85])
        frames.append(
            np.vstack([renderer.render_offscreen(camera_id=cam, width=WIDTH, height=HEIGHT), strip.draw(rows, now, args.start)])
        )
        if terminated or truncated:
            ending = "fell" if env.sim.data.joint("pelvis_ty").qpos[0] < config.env_params.safe_height else "episode limit"
            break
    env.close()

    writer = imageio.get_writer(str(out), fps=round(1 / env.dt), codec="libx264", macro_block_size=None, quality=7)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"wrote {out}: {len(frames) * env.dt:.1f} s, {ending}")


if __name__ == "__main__":
    main()
