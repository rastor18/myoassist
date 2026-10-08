"""The ExoBoot's sensor channels, simulated: what the Pi sends the Jetson for each leg, from the simulated shank and ankle.

On the boot, ``Exo.read_data`` (exoboot.py:313-379) turns the Dephy actuator pack's readings into 8 channels per leg:

* accel x/y/z in g, gravity included, and gyro x/y/z in deg/s, from the 6-axis IMU in the pack on the lateral shank.
  The axes are converted to point forward, up the shank, and laterally outward, so the left leg's frame is left-handed;
* the ankle angle in degrees, plantarflexion-positive, and its velocity, from the encoder
  (``boot_state.AnkleEncoder``, which the boot's other controllers read too).

Here an accelerometer and a gyro sit at the pack's centre on each shank (``add_boot_imus``), in a right-handed site
frame: x forward, y up the shank, z to the right, at the standing keyframe. ``BootSensorFrontEnd`` reads them once per
controller tick and converts them as the boot does. For the left leg's left-handed frame the lateral accelerometer
axis flips, and so do the gyro's forward and vertical components (an angular velocity is a pseudovector); the
sagittal gyro component does not, so forward swing reads positive on both legs, as the boot's heel-strike detector
assumes. The Dephy's resolution (1/8192 g, 1/32.75 deg/s, one encoder click) is applied when ``quantize`` is on.

The MyoAssist leg models are planar, so the out-of-plane channels (accel z, gyro x and y) are ~0 apart from any mount
rotation (``mount_rpy_deg``). The network leans on them: fed the DL validation sessions' logs with those channels as
the planar model gives them, its stance phase is off by RMSE 0.04-0.05 (zeroed: 0.10-0.115). ``OutOfPlaneFilter``
synthesizes them from the in-plane channels instead, with a causal linear filter fit to those logs
(``tools/calibrate_boot_sensors.py --save-filter``); with it, 0.02.
"""

from __future__ import annotations

import json
import math
import re

import mujoco
import numpy as np

from myoassist_utils.exo_ctrl.boot_state import ANKLE_JOINT, AnkleEncoder

PACK_BODY = "DephyExoBoot_L1_exo_1_{side}"
SITE = "boot_imu_{side}"
ACCEL = "boot_imu_accel_{side}"
GYRO = "boot_imu_gyro_{side}"
KNEE_JOINT = "knee_angle_{side}"

ACCEL_LSB_G = 1.0 / 8192.0  # constants.ACCEL_GAIN
GYRO_LSB_DEG_S = 1.0 / 32.75  # constants.GYRO_GAIN

# Per side, (accel, gyro) component signs from the right-handed site frame to the boot's frame: the left leg's frame is
# the site frame with z reversed, which flips the gyro's x and y (an angular velocity is a pseudovector). Each is its
# own inverse.
FRAME_SIGNS = {
    "r": ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
    "l": ((1.0, 1.0, -1.0), (-1.0, -1.0, 1.0)),
}


def _site_frame(model: mujoco.MjModel, data: mujoco.MjData, side: str):
    """World pose of a leg's IMU site at the current state: the pack's geometry centre, axes forward / up the shank / right."""
    body = model.body(PACK_BODY.format(side=side)).id
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
    if not geoms:
        raise ValueError(f"{PACK_BODY.format(side=side)} has no geometry to place the IMU on")
    position = np.mean([data.geom_xpos[g] for g in geoms], axis=0)
    up = data.joint(KNEE_JOINT.format(side=side)).xanchor - data.joint(ANKLE_JOINT.format(side=side)).xanchor
    up = up / np.linalg.norm(up)
    right = np.array([0.0, -1.0, 0.0])  # world y points left on these models
    right = right - right.dot(up) * up
    right = right / np.linalg.norm(right)
    forward = np.cross(up, right)
    return position, np.column_stack([forward, up, right])


def add_boot_imus(xml: str, *, sides=("r", "l")) -> str:
    """The composed model with an IMU site, accelerometer and gyro in each leg's actuator pack.

    Inserted as text, so the rest of the model is unchanged byte for byte: sites and sensors add no mass or contact, and
    the new sensors go after the existing ones, so every existing sensor keeps its address.
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    sensors = []
    for side in sides:
        body = model.body(PACK_BODY.format(side=side)).id
        position, rotation = _site_frame(model, data, side)
        body_rotation = data.xmat[body].reshape(3, 3)
        local_pos = body_rotation.T @ (position - data.xpos[body])
        local_quat = np.zeros(4)
        mujoco.mju_mat2Quat(local_quat, (body_rotation.T @ rotation).ravel())
        site = (
            f'<site name="{SITE.format(side=side)}" pos="{" ".join(repr(float(v)) for v in local_pos)}" '
            f'quat="{" ".join(repr(float(v)) for v in local_quat)}" size="0.005"/>'
        )
        opening = re.search(rf'<body\b[^>]*\bname="{re.escape(PACK_BODY.format(side=side))}"[^>]*(?<!/)>', xml)
        if opening is None:
            raise ValueError(f"no <body name={PACK_BODY.format(side=side)!r}> with children in the model")
        xml = xml[: opening.end()] + site + xml[opening.end() :]
        sensors.append(f'<accelerometer name="{ACCEL.format(side=side)}" site="{SITE.format(side=side)}"/>')
        sensors.append(f'<gyro name="{GYRO.format(side=side)}" site="{SITE.format(side=side)}"/>')
    closing = xml.rfind("</sensor>")
    if closing < 0:
        raise ValueError("the model has no <sensor> section")
    return xml[:closing] + "".join(sensors) + xml[closing:]


def rotation_from_rpy_deg(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """The IMU's orientation in the site frame, R = Rz(yaw) Ry(pitch) Rx(roll), about the site's own axes: roll about x
    (forward), pitch about y (up the shank), yaw about z (to the right, so yaw is the sagittal tilt)."""
    r, p, y = (math.radians(a) for a in (roll, pitch, yaw))
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def rpy_deg_from_rotation(rotation: np.ndarray) -> tuple[float, float, float]:
    """The inverse of ``rotation_from_rpy_deg``, with pitch in [-90, 90] deg."""
    r = np.asarray(rotation, dtype=float)
    roll = math.atan2(r[2, 1], r[2, 2])
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    yaw = math.atan2(r[1, 0], r[0, 0])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class OutOfPlaneFilter:
    """One leg's out-of-plane IMU components (site-frame accel z, gyro x and y) predicted from the in-plane ones' last
    0.5 s: a causal FIR with an intercept, fit by ridge regression on boot logs.

    Inputs are in the right-handed site frame (``FRAME_SIGNS`` and the mount undone), in the boot's units. By default it
    reads the site-frame gyro z and the ankle velocity: on the DL validation sessions those do as well as all five
    sagittal channels held out (stance-phase RMSE ~0.02), and they are what the sim reproduces best. The ankle angle
    would carry an offset the sim does not share (the reference gait reads 4-9 deg below the boot's), and gyro z alone
    puts heel strikes ~29 ms early.

    The lags count samples at ``rate_hz``, so a filter only applies at the rate it was fit at. ``predict`` runs over a
    whole recording, ``step`` one sample at a time; both treat the time before the first sample as that sample.
    """

    LAGS = np.arange(0, 88, 3)  # samples back at 175 Hz: 0 to 0.49 s
    ALL_FEATURES = ("accel_x", "accel_y", "gyro_z", "ankle_angle", "ankle_velocity")
    FEATURES = ("gyro_z", "ankle_velocity")
    RATE_HZ = 175.0

    def __init__(
        self,
        coefficients: np.ndarray,
        ankle_mean: float = 0.0,
        features=FEATURES,
        *,
        lags=LAGS,
        rate_hz: float = RATE_HZ,
        ridge: float = math.nan,
        loo_rmse=(),
    ):
        """``ridge`` and ``loo_rmse`` (the network's stance-phase RMSE on each session left out of the fit) are only
        carried along, for whoever reads a saved filter."""
        self.features = tuple(features)
        if (
            not self.features
            or len(set(self.features)) != len(self.features)
            or not set(self.features) <= set(self.ALL_FEATURES)
        ):
            raise ValueError(f"features must be distinct names from {self.ALL_FEATURES}, got {self.features}")
        self.lags = np.asarray(lags)
        if self.lags.ndim != 1 or not len(self.lags) or self.lags.dtype.kind not in "iu" or self.lags[0] < 0:
            raise ValueError(f"lags must be non-negative integers, got {lags!r}")
        if np.any(np.diff(self.lags) <= 0):
            raise ValueError(f"lags must increase, got {self.lags.tolist()}")
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        shape = (1 + len(self.lags) * len(self.features), 3)
        if self.coefficients.size and self.coefficients.shape != shape:  # empty: a probe for fit's design matrix
            raise ValueError(
                f"coefficients have shape {self.coefficients.shape}; {len(self.features)} features at "
                f"{len(self.lags)} lags need {shape}"
            )
        if not np.all(np.isfinite(self.coefficients)):
            raise ValueError("coefficients are not finite")
        if not (math.isfinite(rate_hz) and rate_hz > 0):
            raise ValueError(f"rate_hz must be positive, got {rate_hz}")
        self.ankle_mean = float(ankle_mean)
        self.rate_hz = float(rate_hz)
        self.ridge = float(ridge)
        self.loo_rmse = tuple(float(v) for v in loo_rmse)
        self._history = np.zeros((int(self.lags[-1]) + 1, len(self.features)))
        self._row = np.ones(shape[0])
        self.reset()

    # --- offline ------------------------------------------------------------------------------------------------------

    def _feature_columns(self, accel, gyro, x) -> np.ndarray:
        """[n, feature]. ``x`` is the boot's 8 channels, for the ankle's two."""
        available = {
            "accel_x": lambda: accel[:, 0],
            "accel_y": lambda: accel[:, 1],
            "gyro_z": lambda: gyro[:, 2],
            "ankle_angle": lambda: x[:, 6] - self.ankle_mean,
            "ankle_velocity": lambda: x[:, 7],
        }
        return np.column_stack([available[f]() for f in self.features])

    def _design(self, accel, gyro, x) -> np.ndarray:
        sagittal = self._feature_columns(accel, gyro, x)
        n = len(sagittal)
        columns = [np.ones(n)]
        for lag in self.lags:
            lagged = np.vstack([np.repeat(sagittal[:1], lag, axis=0), sagittal[: n - lag]]) if lag else sagittal
            columns.extend(lagged.T)
        return np.column_stack(columns)

    @classmethod
    def fit(cls, examples, *, features=FEATURES, ridge: float = 1e-3, lags=LAGS, rate_hz: float = RATE_HZ) -> OutOfPlaneFilter:
        """``examples``: (accel, gyro, x, rows) per recording, accel and gyro already in the site frame; ``rows`` picks
        the rows to fit on. ``ridge`` is relative to the design's mean diagonal."""
        ankle_mean = float(np.mean(np.concatenate([x[rows, 6] for _, _, x, rows in examples])))
        probe = cls(np.zeros(0), ankle_mean, features, lags=lags, rate_hz=rate_hz)
        design = np.vstack([probe._design(a, g, x)[rows] for a, g, x, rows in examples])
        target = np.vstack([np.column_stack([a[rows, 2], g[rows, 0], g[rows, 1]]) for a, g, x, rows in examples])
        lam = ridge * np.trace(design.T @ design) / design.shape[1]
        coefficients = np.linalg.solve(design.T @ design + lam * np.eye(design.shape[1]), design.T @ target)
        return cls(coefficients, ankle_mean, features, lags=lags, rate_hz=rate_hz, ridge=ridge)

    def predict(self, accel, gyro, x) -> np.ndarray:
        """[n, 3]: accel z, gyro x, gyro y on every row of a recording."""
        return self._design(accel, gyro, x) @ self.coefficients

    # --- streaming ----------------------------------------------------------------------------------------------------

    def reset(self) -> None:
        self._count = 0

    def step(self, accel, gyro, ankle_angle: float, ankle_velocity: float) -> tuple[float, float, float]:
        """The next sample's (accel z, gyro x, gyro y), from its site-frame ``accel`` and ``gyro`` and the ankle."""
        available = {
            "accel_x": lambda: accel[0],
            "accel_y": lambda: accel[1],
            "gyro_z": lambda: gyro[2],
            "ankle_angle": lambda: ankle_angle - self.ankle_mean,
            "ankle_velocity": lambda: ankle_velocity,
        }
        sample = [float(available[f]()) for f in self.features]
        size = len(self._history)
        if self._count == 0:
            self._history[:] = sample  # before the first sample, as ``predict`` pads
        position = self._count % size
        self._history[position] = sample
        self._count += 1
        self._row[1:] = self._history[(position - self.lags) % size].ravel()  # the design row: lag-major, as _design
        out = self._row @ self.coefficients
        return float(out[0]), float(out[1]), float(out[2])

    # --- files --------------------------------------------------------------------------------------------------------

    def _metadata(self) -> dict:
        return dict(
            features=list(self.features),
            ankle_mean=self.ankle_mean,
            rate_hz=self.rate_hz,
            ridge=self.ridge,
            loo_rmse=list(self.loo_rmse),
        )


OUT_OF_PLANE_FORMAT = "exoboot out-of-plane filter, 1"


def save_out_of_plane_filters(path, filters) -> None:
    """Write each side's filter (``{"r": ..., "l": ...}``) to one ``.npz``."""
    if not filters or not set(filters) <= set(FRAME_SIGNS):
        raise ValueError(f"need filters by side ('r', 'l'), got {sorted(filters)}")
    arrays = {}
    for side, f in filters.items():
        arrays[f"{side}/coefficients"] = f.coefficients
        arrays[f"{side}/lags"] = f.lags.astype(np.int64)
    metadata = dict(format=OUT_OF_PLANE_FORMAT, sides={side: f._metadata() for side, f in filters.items()})
    np.savez(path, metadata=np.array(json.dumps(metadata)), **arrays)


def load_out_of_plane_filters(path) -> dict[str, OutOfPlaneFilter]:
    """Each side's filter from a ``save_out_of_plane_filters`` file; anything else is refused."""
    with np.load(path, allow_pickle=False) as npz:
        if "metadata" not in npz.files:
            raise ValueError(f"{path} is not an out-of-plane filter file (no metadata)")
        try:
            metadata = json.loads(str(npz["metadata"]))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: unreadable metadata ({e})") from None
        if not isinstance(metadata, dict) or metadata.get("format") != OUT_OF_PLANE_FORMAT:
            raise ValueError(f"{path} is not an out-of-plane filter file (format {metadata.get('format')!r})")
        sides = metadata.get("sides")
        if not isinstance(sides, dict) or not sides or not set(sides) <= set(FRAME_SIGNS):
            raise ValueError(f"{path}: need filters by side ('r', 'l'), got {sides!r}")
        filters = {}
        for side, meta in sides.items():
            missing = sorted({f"{side}/coefficients", f"{side}/lags"} - set(npz.files))
            missing += sorted({"features", "rate_hz"} - set(meta))
            if missing:
                raise ValueError(f"{path}: side {side!r} lacks {missing}")
            filters[side] = OutOfPlaneFilter(
                npz[f"{side}/coefficients"],
                meta.get("ankle_mean", 0.0),
                meta["features"],
                lags=npz[f"{side}/lags"],
                rate_hz=float(meta["rate_hz"]),
                ridge=float(meta.get("ridge", math.nan)),
                loo_rmse=meta.get("loo_rmse", ()),
            )
    return filters


class BootSensorFrontEnd:
    """One leg's 8 channels (``gait_net.INPUT_CHANNELS``) in the boot's units and signs, read once per controller tick.

    ``standing_angle_deg`` is what the boot's ankle angle reads at the standing keyframe, the session's
    ``<SIDE>_STANDING_ANGLE``: it fixes the offset the boot adds to its encoder angle. ``mount_rpy_deg`` rotates the
    simulated IMU in its own frame, for a pack that does not sit square on the shank.

    With ``out_of_plane`` (this leg's ``OutOfPlaneFilter``), the site-frame accel z and gyro x and y the planar model
    gives are replaced by the filter's prediction before the mount, signs and quantization, as if the shank moved out of
    the plane the way the boot's did.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        side: str,
        *,
        standing_angle_deg: float,
        sample_rate_hz: float = 175.0,
        mount_rpy_deg=(0.0, 0.0, 0.0),
        quantize: bool = True,
        out_of_plane: OutOfPlaneFilter | None = None,
    ):
        if side not in FRAME_SIGNS:
            raise ValueError(f"side must be 'r' or 'l', got {side!r}")
        if out_of_plane is not None:
            if out_of_plane.rate_hz != sample_rate_hz:
                raise ValueError(
                    f"the out-of-plane filter was fit at {out_of_plane.rate_hz:g} Hz and its lags count those samples; "
                    f"this front end samples at {sample_rate_hz:g} Hz"
                )
            if "ankle_angle" in out_of_plane.features:
                raise ValueError(
                    "an out-of-plane filter reading the ankle angle would carry the sim's ankle offset into the "
                    "synthesized channels; fit one without it"
                )
        self.side = side
        self.quantize = quantize
        self.out_of_plane = out_of_plane

        def sensor_adr(name):
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sensor_id < 0:
                raise KeyError(f"no sensor {name!r}: compose the model through boot_sensors.add_boot_imus")
            return int(model.sensor_adr[sensor_id])

        self._accel = sensor_adr(ACCEL.format(side=side))
        self._gyro = sensor_adr(GYRO.format(side=side))
        self.ankle = AnkleEncoder(
            model, side, standing_angle_deg=standing_angle_deg, sample_rate_hz=sample_rate_hz, quantize=quantize
        )
        self._g = float(np.linalg.norm(model.opt.gravity))
        mount = rotation_from_rpy_deg(*mount_rpy_deg).T  # site-frame vectors -> the rotated IMU's frame
        accel_sign, gyro_sign = FRAME_SIGNS[side]
        self._accel_matrix = [[accel_sign[i] * float(mount[i][j]) for j in range(3)] for i in range(3)]
        self._gyro_matrix = [[gyro_sign[i] * float(mount[i][j]) for j in range(3)] for i in range(3)]
        self.reset()

    def reset(self) -> None:
        self.ankle.reset()
        if self.out_of_plane is not None:
            self.out_of_plane.reset()

    @staticmethod
    def _quantize(value: float, lsb: float) -> float:
        return round(value / lsb) * lsb

    def sample(self, sim) -> list[float]:
        ankle = self.ankle.read(sim)
        sensordata = sim.data.sensordata
        accel = [float(sensordata[self._accel + i]) / self._g for i in range(3)]
        gyro = [math.degrees(float(sensordata[self._gyro + i])) for i in range(3)]
        if self.out_of_plane is not None:
            accel[2], gyro[0], gyro[1] = self.out_of_plane.step(accel, gyro, *ankle)
        a = [sum(row[j] * accel[j] for j in range(3)) for row in self._accel_matrix]
        w = [sum(row[j] * gyro[j] for j in range(3)) for row in self._gyro_matrix]
        if self.quantize:
            a = [self._quantize(v, ACCEL_LSB_G) for v in a]
            w = [self._quantize(v, GYRO_LSB_DEG_S) for v in w]
        return [*a, *w, *ankle]
