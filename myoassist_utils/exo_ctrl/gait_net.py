"""The ExoBoot's 4-headed gait network (the DL task), rebuilt from its TensorRT engine: full-window and streaming passes.

On the boot the Raspberry Pi sends each leg's 8 sensor channels to a Jetson on every 175 Hz tick. The Jetson keeps the last
200 samples per leg in a zero-filled shift buffer, newest last, and runs this network on it. Four independent branches,
one per output, each:

    Conv1D 8->44 (kernel 8, dilation 6) -> +bias, ReLU, BatchNorm
    Conv1D 44->44 (kernel 8, dilation 6) -> +bias, ReLU, BatchNorm
    Conv1D 44->44 (kernel 8, dilation 6) -> +bias, ReLU
    Conv1D 44->44 (kernel 74)            -> +bias, ReLU      (200 -> 158 -> 116 -> 74 -> 1 samples, all "valid")
    Dense 44->23, then 23->23 three times -> +bias, ReLU each
    Dense 23->1                           -> through a sigmoid for stance_swing; linear for the other three

BatchNorm is kept folded into a per-channel scale and shift, which is all the engine stores. ``tools/recover_gait_net.py``
extracts the weights into the ``.npz`` this module reads.

Inputs are raw, in the units and signs the boot sends: accel x/y/z in g, gyro x/y/z in deg/s, ankle angle in deg
(plantarflexion-positive, with the standing offset), ankle velocity in deg/s. The network normalizes them itself.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np

HEADS = ("velocity", "stance_swing", "ramp", "stance_phase")
SIGMOID_HEADS = frozenset({"stance_swing"})
INPUT_CHANNELS = (
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "ankle_angle",
    "ankle_velocity",
)
WINDOW = 200
TAPS, DILATION, SPAN = (
    8,
    6,
    43,
)  # a dilated conv reads SPAN = (TAPS - 1) * DILATION + 1 consecutive positions
WIDTH, HIDDEN, LONG_TAPS = 44, 23, 74

# Per head, the parameter arrays and their shapes. Kernels are [out, in, tap] for convolutions and [out, in] for dense
# layers, the TensorRT/ONNX order; taps run oldest to newest, so conv output position j reads inputs j + DILATION * s.
PARAM_SHAPES = {
    "conv1/kernel": (WIDTH, len(INPUT_CHANNELS), TAPS),
    "conv1/bias": (WIDTH,),
    "bn1/scale": (WIDTH,),
    "bn1/shift": (WIDTH,),
    "conv2/kernel": (WIDTH, WIDTH, TAPS),
    "conv2/bias": (WIDTH,),
    "bn2/scale": (WIDTH,),
    "bn2/shift": (WIDTH,),
    "conv3/kernel": (WIDTH, WIDTH, TAPS),
    "conv3/bias": (WIDTH,),
    "conv4/kernel": (WIDTH, WIDTH, LONG_TAPS),
    "conv4/bias": (WIDTH,),
    "dense1/kernel": (HIDDEN, WIDTH),
    "dense1/bias": (HIDDEN,),
    "dense2/kernel": (HIDDEN, HIDDEN),
    "dense2/bias": (HIDDEN,),
    "dense3/kernel": (HIDDEN, HIDDEN),
    "dense3/bias": (HIDDEN,),
    "dense4/kernel": (HIDDEN, HIDDEN),
    "dense4/bias": (HIDDEN,),
    "out/kernel": (HIDDEN,),
    "out/bias": (),
}
DENSE = ("dense1", "dense2", "dense3", "dense4")

Weights = Mapping[str, Mapping[str, np.ndarray]]


def validate_weights(weights: Weights) -> None:
    """Every head has every parameter, at its shape, and finite."""
    if set(weights) != set(HEADS):
        raise ValueError(f"need weights for heads {HEADS}, got {sorted(weights)}")
    for head in HEADS:
        params = weights[head]
        if set(params) != set(PARAM_SHAPES):
            missing, extra = (
                set(PARAM_SHAPES) - set(params),
                set(params) - set(PARAM_SHAPES),
            )
            raise ValueError(f"{head}: missing {sorted(missing)}, unexpected {sorted(extra)}")
        for name, shape in PARAM_SHAPES.items():
            value = np.asarray(params[name])
            if value.shape != shape:
                raise ValueError(f"{head}/{name} has shape {value.shape}, expected {shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{head}/{name} is not finite")


def save_weights(path, weights: Weights, metadata: Mapping | None = None, *, dtype=np.float64) -> None:
    """Write the weights to ``.npz``. A narrower ``dtype`` is refused unless it holds every value exactly."""
    validate_weights(weights)
    arrays = {}
    for head in HEADS:
        for name in PARAM_SHAPES:
            value = np.asarray(weights[head][name], dtype=np.float64)
            stored = value.astype(dtype)
            if not np.array_equal(stored.astype(np.float64), value):
                raise ValueError(f"{head}/{name} does not fit {np.dtype(dtype).name} exactly")
            arrays[f"{head}/{name}"] = stored
    np.savez(path, metadata=np.array(json.dumps(dict(metadata or {}))), **arrays)


def load_weights(path) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """The weights in a ``.npz`` from ``save_weights``, and its metadata."""
    with np.load(path, allow_pickle=False) as npz:
        weights = {head: {name: npz[f"{head}/{name}"].astype(np.float64) for name in PARAM_SHAPES} for head in HEADS}
        metadata = json.loads(str(npz["metadata"])) if "metadata" in npz.files else {}
    validate_weights(weights)
    return weights, metadata


def _check_heads(heads) -> tuple[str, ...]:
    heads = tuple(heads)
    if not heads or len(set(heads)) != len(heads) or not set(heads) <= set(HEADS):
        raise ValueError(f"heads must be distinct names from {HEADS}, got {heads}")
    return heads


class _Stacked:
    """The chosen branches' parameters stacked on a leading branch axis, in one dtype, plus their matmul-ready forms."""

    def __init__(self, weights: Weights, heads, dtype):
        validate_weights(weights)
        self.heads = heads

        def stack(name):
            return np.stack([np.asarray(weights[head][name], dtype=dtype) for head in heads])

        n = len(heads)
        self.k1, self.b1, self.m1, self.a1 = (
            stack("conv1/kernel"),
            stack("conv1/bias"),
            stack("bn1/scale"),
            stack("bn1/shift"),
        )
        self.k2, self.b2, self.m2, self.a2 = (
            stack("conv2/kernel"),
            stack("conv2/bias"),
            stack("bn2/scale"),
            stack("bn2/shift"),
        )
        self.k3, self.b3 = stack("conv3/kernel"), stack("conv3/bias")
        self.k4, self.b4 = stack("conv4/kernel"), stack("conv4/bias")
        self.dense = [(stack(f"{d}/kernel"), stack(f"{d}/bias")) for d in DENSE]
        self.ko, self.bo = stack("out/kernel"), stack("out/bias")
        self.sigmoid = [head in SIGMOID_HEADS for head in heads]
        # [branch * out, in * tap] and [branch, out, in * tap]: a dilated conv's new position, or the 74-tap conv, as
        # one matmul call over every branch and stream at once.
        self.k1m = self.k1.reshape(n * WIDTH, -1)
        self.k2m, self.k3m, self.k4m = (k.reshape(n, WIDTH, -1) for k in (self.k2, self.k3, self.k4))


def _taps(x: np.ndarray, t_out: int) -> np.ndarray:
    """[..., c, t] -> [..., c, TAPS, t_out]: what each of the t_out output positions of a dilated conv reads."""
    return np.stack([x[..., s * DILATION : s * DILATION + t_out] for s in range(TAPS)], axis=-2)


def _sigmoid(y):
    return 1.0 / (1.0 + np.exp(-y))


class FullWindowGaitNet:
    """The network on whole windows, as the Jetson runs it: [..., 200, 8] -> one array [...] per head. The reference.

    ``heads`` picks which branches to evaluate; each is independent of the others.
    """

    def __init__(self, weights: Weights, *, heads=HEADS, dtype=np.float64):
        self.heads = _check_heads(heads)
        self.dtype = dtype
        self._p = _Stacked(weights, self.heads, dtype)

    def __call__(self, windows) -> dict[str, np.ndarray]:
        p = self._p
        x = np.swapaxes(np.asarray(windows, dtype=self.dtype), -1, -2)  # [..., 8, 200]
        if x.shape[-2:] != (len(INPUT_CHANNELS), WINDOW):
            raise ValueError(f"windows must be [..., {WINDOW}, {len(INPUT_CHANNELS)}], got {np.shape(windows)}")
        h = np.einsum("bkcs,...cst->...bkt", p.k1, _taps(x, WINDOW - SPAN + 1))
        h = np.maximum(h + p.b1[..., None], 0.0) * p.m1[..., None] + p.a1[..., None]
        h = np.einsum("bkcs,...bcst->...bkt", p.k2, _taps(h, h.shape[-1] - SPAN + 1))
        h = np.maximum(h + p.b2[..., None], 0.0) * p.m2[..., None] + p.a2[..., None]
        h = np.einsum("bkcs,...bcst->...bkt", p.k3, _taps(h, h.shape[-1] - SPAN + 1))
        h = np.maximum(h + p.b3[..., None], 0.0)
        h = np.maximum(np.einsum("bkcs,...bcs->...bk", p.k4, h) + p.b4, 0.0)
        for kernel, bias in p.dense:
            h = np.maximum(np.einsum("bkc,...bc->...bk", kernel, h) + bias, 0.0)
        y = np.einsum("bc,...bc->...b", p.ko, h) + p.bo
        return {head: _sigmoid(y[..., i]) if sig else y[..., i] for i, (head, sig) in enumerate(zip(self.heads, p.sigmoid))}


class StreamingGaitNet:
    """``FullWindowGaitNet`` on a zero-filled 200-sample shift buffer, advanced one sample per ``step``.

    Each layer keeps the outputs it already computed, so a step adds one new position to each dilated conv instead of
    recomputing the window; only the 74-tap conv and the dense layers see their whole input again. That is about a
    twentieth of the work of the full window, in the same arithmetic, so the two agree to rounding.

    Runs several independent streams at once (the two legs): ``samples`` is [n_streams, 8]. ``reset`` puts every cache
    at the activations of an all-zero window, which is exactly the Jetson's buffer before its first sample. ``heads``
    picks which branches to run: the boot's torque needs only stance_phase and stance_swing.

    Every contraction is one matmul call, ``[branch, out, in] @ [stream, branch, in, 1]``, which numpy runs as a
    matrix-vector product per stream and branch. With two streams that takes about half the time of a matrix product
    over the streams, which BLAS packs first (the 74-tap conv's weights are 0.6 MB per branch).
    """

    def __init__(self, weights: Weights, *, n_streams: int = 1, heads=HEADS, dtype=np.float64):
        if n_streams < 1:
            raise ValueError(f"n_streams must be >= 1, got {n_streams}")
        self.heads = _check_heads(heads)
        self.n_streams = n_streams
        self.dtype = dtype
        self._p = p = _Stacked(weights, self.heads, dtype)
        n_branch = len(self.heads)
        # The activations of an all-zero window: every position of every layer is the same. Shapes [branch, 44].
        zero1 = np.maximum(p.b1, 0.0) * p.m1 + p.a1
        zero2 = np.maximum(np.einsum("bkcs,bc->bk", p.k2, zero1) + p.b2, 0.0) * p.m2 + p.a2
        zero3 = np.maximum(np.einsum("bkcs,bc->bk", p.k3, zero2) + p.b3, 0.0)
        self._zero = (zero1, zero2, zero3)
        # Caches are [stream, (branch,) channel, position], so each stream's input to a contraction is contiguous.
        self._x = np.zeros((n_streams, len(INPUT_CHANNELS), SPAN), dtype)
        self._h1 = np.zeros((n_streams, n_branch, WIDTH, SPAN), dtype)
        self._h2 = np.zeros((n_streams, n_branch, WIDTH, SPAN), dtype)
        self._h3 = np.zeros((n_streams, n_branch, WIDTH, LONG_TAPS), dtype)
        self._ko = p.ko[:, None, :]
        self.reset()

    def reset(self, streams=None) -> None:
        """Back to an all-zero window, for every stream or the given ones."""
        s = slice(None) if streams is None else np.atleast_1d(streams)
        zero1, zero2, zero3 = self._zero
        self._x[s] = 0.0
        self._h1[s] = zero1[:, :, None]
        self._h2[s] = zero2[:, :, None]
        self._h3[s] = zero3[:, :, None]

    def snapshot(self) -> tuple[np.ndarray, ...]:
        """A copy of the buffer state, to try something and come back with ``restore``."""
        return tuple(cache.copy() for cache in (self._x, self._h1, self._h2, self._h3))

    def restore(self, snapshot: tuple[np.ndarray, ...]) -> None:
        for cache, saved in zip((self._x, self._h1, self._h2, self._h3), snapshot, strict=True):
            cache[...] = saved

    @staticmethod
    def _push(cache: np.ndarray, column: np.ndarray) -> None:
        cache[..., :-1] = cache[..., 1:]
        cache[..., -1] = column

    def step(self, samples) -> dict[str, np.ndarray]:
        """Append one sample per stream ([n_streams, 8], channels as ``INPUT_CHANNELS``); one array [n_streams] per head."""
        p, n = self._p, self.n_streams
        n_branch = len(self.heads)
        self._push(self._x, np.asarray(samples, dtype=self.dtype).reshape(n, len(INPUT_CHANNELS)))
        h = (p.k1m @ self._x[:, :, ::DILATION].reshape(n, -1, 1)).reshape(n, n_branch, WIDTH)
        self._push(self._h1, np.maximum(h + p.b1, 0.0) * p.m1 + p.a1)
        h = (p.k2m @ self._h1[..., ::DILATION].reshape(n, n_branch, -1, 1))[..., 0]
        self._push(self._h2, np.maximum(h + p.b2, 0.0) * p.m2 + p.a2)
        h = (p.k3m @ self._h2[..., ::DILATION].reshape(n, n_branch, -1, 1))[..., 0]
        self._push(self._h3, np.maximum(h + p.b3, 0.0))
        h = np.maximum((p.k4m @ self._h3.reshape(n, n_branch, -1, 1))[..., 0] + p.b4, 0.0)
        for kernel, bias in p.dense:
            h = np.maximum((kernel @ h[..., None])[..., 0] + bias, 0.0)
        y = (self._ko @ h[..., None])[..., 0, 0] + p.bo  # [n, branch]
        return {head: _sigmoid(y[:, i]) if sig else y[:, i] for i, (head, sig) in enumerate(zip(self.heads, p.sigmoid))}
