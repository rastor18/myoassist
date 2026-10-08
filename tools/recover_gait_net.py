"""Recover the ExoBoot's 4-headed gait network from the Jetson's TensorRT engine, as weights for ``gait_net.py``.

The original Keras model is not available, but the engine (``jetson_code/4_headed/final_model_all.trt``, TensorRT 8.5 for
Orin) keeps every weight raw, and it was built with detailed layer information, which names each layer and gives its
weight count, precision and tactic -- and the tactic fixes the memory layout. What the engine shows:

* The four first convolutions are fused into one 8->176 FP32 convolution (KCRS), in the order conv1d_8, conv1d_4,
  conv1d_12, conv1d. Every later group of four layers runs in the same branch order, and the branches feed
  velocity_output, stance_swing_output, ramp_output and stance_phase_output.
* Bias, ReLU and the folded BatchNorm of each first and second convolution run as one pointwise op on three [44]
  constants: bias, scale, shift. The third convolution's bias is a fourth constant.
* The second and third convolutions run in FP16 with KRSC weights, input channels padded from 44 to 48 with zeros.
* The 74-tap convolutions and the dense layers are FP32, weights then bias. Dense layers run as 1x1 convolutions,
  kernel [out, in].
* Weights sit in execution order on 256-byte slots, after a 225-byte header: the 28 constants, the fused first
  convolution, the eight FP16 convolutions, the four 74-tap ones, then the dense layers and the heads.

The offsets below are this engine's, so the engine is checked by hash before anything is read, and every assumption
about the layout is checked against the bytes: the FP16 padding is exactly zero, each BatchNorm scale is positive, the
layer names appear in the order assumed, and so on. On the DL validation sessions the recovered network reproduces the
Jetson's logged replies to within its FP16 arithmetic (``tools/replay_dl_session.py``).

    python tools/recover_gait_net.py "<...>/jetson_code/4_headed/final_model_all.trt" --out gait_net_4headed.npz
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib

import numpy as np

from myoassist_utils.exo_ctrl.gait_net import (
    HEADS,
    HIDDEN,
    LONG_TAPS,
    TAPS,
    WIDTH,
    save_weights,
)

ENGINE_SHA256 = "66958ca37901d8b64d782c66ec60a2c0d0cb8c284a294e33450e996a9824cd88"
N_INPUT = 8

# Branch order in the engine, with each branch's Keras layer names: (conv1d indices, batch_normalization indices,
# dense indices, output name). HEADS lists the heads in this same order.
BRANCHES = {
    "velocity": ((8, 9, 10, 11), (4, 5), (8, 9, 10, 11), "velocity_output"),
    "stance_swing": ((4, 5, 6, 7), (2, 3), (4, 5, 6, 7), "stance_swing_output"),
    "ramp": ((12, 13, 14, 15), (6, 7), (12, 13, 14, 15), "ramp_output"),
    "stance_phase": ((0, 1, 2, 3), (0, 1), (0, 1, 2, 3), "stance_phase_output"),
}

HEADER, SLOT = 225, 256
CONSTANTS = [HEADER + 2 * SLOT * i for i in range(28)]  # 7 per branch: bias1, scale1, shift1, bias2, scale2, shift2, bias3
CONV1 = 14305
CONV1_ONES = CONV1 + 176 * N_INPUT * TAPS * 4  # 176 floats of exactly 1.0 follow it, unused here; checked as a landmark
FP16_CONVS = [
    60129,
    97249,
    134113,
    170977,
    207841,
    244705,
    281569,
    318433,
]  # second convs then third, branch order
CONV4 = [355297, 928865, 1502177, 2075489]
DENSE1 = [
    (2648801, 2652897),
    (2653153, 2657505),
    (2657761, 2662113),
    (2662369, 2666721),
]  # (kernel, bias) per branch
DENSE23 = [(2666977 + 2560 * i, 2666977 + 2560 * i + 2304) for i in range(12)]  # 3 deep x 4 branches
HEAD_OUT = [(2697697 + 512 * i, 2697953 + 512 * i) for i in range(4)]
PADDED_CHANNELS = 48


def _keras_name(kind: str, index: int) -> str:
    return f"model/{kind}" if index == 0 else f"model/{kind}_{index}"


def check_layer_order(raw: bytes) -> None:
    """The layer names appear in the engine in the branch order the offsets assume.

    Only the layer records count: they are in execution order and begin with the first fused pointwise op. An earlier
    table also lists tensor names, in no particular order.
    """
    records = raw.find(b"PWN(PWN(")
    if records < 0:
        raise ValueError("no fused pointwise layer records found; this is not the engine layout assumed")

    def first(name: str, start: int = records) -> int:
        at = raw.find(name.encode(), start)
        if at < 0:
            raise ValueError(f"layer {name!r} not found in the engine's layer records")
        return at

    fused = " || ".join(f"{_keras_name('conv1d', BRANCHES[h][0][0])}/Conv1D" for h in HEADS)
    first(fused, start=0)
    for depth in range(1, 4):
        positions = [first(f"{_keras_name('conv1d', BRANCHES[h][0][depth])}/Conv1D") for h in HEADS]
        if positions != sorted(positions):
            raise ValueError(f"conv depth {depth + 1} is not in branch order {HEADS}")
    for depth in range(4):
        positions = [first(f"{_keras_name('dense', BRANCHES[h][2][depth])}/Tensordot/MatMul") for h in HEADS]
        if positions != sorted(positions):
            raise ValueError(f"dense depth {depth + 1} is not in branch order {HEADS}")
    positions = [first(f"model/{BRANCHES[h][3]}/Tensordot/MatMul") for h in HEADS]
    if positions != sorted(positions):
        raise ValueError(f"output heads are not in branch order {HEADS}")
    # Each head reads the last dense layer of its own branch.
    for head in HEADS:
        out = first(f"model/{BRANCHES[head][3]}/Tensordot/Reshape")
        feed = raw.rfind(b"/Relu:0", 0, out)
        start = raw.rfind(b"model/", 0, feed)
        expected = f"{_keras_name('dense', BRANCHES[head][2][3])}/Relu:0".encode()
        if raw[start : feed + len(b"/Relu:0")] != expected:
            raise ValueError(f"{head} does not read {expected.decode()}")


def read_engine(path: str | pathlib.Path, *, check_hash: bool = True) -> dict[str, dict[str, np.ndarray]]:
    raw = pathlib.Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if check_hash and digest != ENGINE_SHA256:
        raise ValueError(
            f"{path} has sha256 {digest}, not the engine these offsets were read from ({ENGINE_SHA256}); "
            "a different build lays its weights out differently"
        )
    check_layer_order(raw)

    def f32(offset, count):
        return np.frombuffer(raw, dtype="<f4", count=count, offset=offset).astype(np.float64)

    def f16(offset, count):
        return np.frombuffer(raw, dtype="<f2", count=count, offset=offset).astype(np.float64)

    if not np.all(f32(CONV1_ONES, 176) == 1.0):
        raise ValueError("the 176 ones after the fused first convolution are missing; the layout is not the one assumed")
    conv1 = f32(CONV1, 176 * N_INPUT * TAPS).reshape(176, N_INPUT, TAPS)  # KCRS with R = 1

    weights = {}
    for b, head in enumerate(HEADS):
        bias1, scale1, shift1, bias2, scale2, shift2, bias3 = (f32(CONSTANTS[7 * b + i], WIDTH) for i in range(7))
        for name, scale in (("bn1", scale1), ("bn2", scale2)):
            if not np.all(scale > 0):
                raise ValueError(f"{head}/{name} scale is not all positive, so the constants are not where assumed")
        convs = []
        for offset in (FP16_CONVS[b], FP16_CONVS[4 + b]):
            krsc = f16(offset, WIDTH * TAPS * PADDED_CHANNELS).reshape(WIDTH, TAPS, PADDED_CHANNELS)
            if not np.all(krsc[:, :, WIDTH:] == 0):
                raise ValueError(f"{head}: FP16 conv at {offset} has non-zero channel padding")
            convs.append(np.ascontiguousarray(krsc[:, :, :WIDTH].transpose(0, 2, 1)))  # -> [out, in, tap]
        n4 = WIDTH * WIDTH * LONG_TAPS
        params = {
            "conv1/kernel": conv1[WIDTH * b : WIDTH * (b + 1)],
            "conv1/bias": bias1,
            "bn1/scale": scale1,
            "bn1/shift": shift1,
            "conv2/kernel": convs[0],
            "conv2/bias": bias2,
            "bn2/scale": scale2,
            "bn2/shift": shift2,
            "conv3/kernel": convs[1],
            "conv3/bias": bias3,
            "conv4/kernel": f32(CONV4[b], n4).reshape(WIDTH, WIDTH, LONG_TAPS),
            "conv4/bias": f32(CONV4[b] + 4 * n4, WIDTH),
            "dense1/kernel": f32(DENSE1[b][0], HIDDEN * WIDTH).reshape(HIDDEN, WIDTH),
            "dense1/bias": f32(DENSE1[b][1], HIDDEN),
        }
        for depth in range(3):
            kernel, bias = DENSE23[4 * depth + b]
            params[f"dense{depth + 2}/kernel"] = f32(kernel, HIDDEN * HIDDEN).reshape(HIDDEN, HIDDEN)
            params[f"dense{depth + 2}/bias"] = f32(bias, HIDDEN)
        params["out/kernel"] = f32(HEAD_OUT[b][0], HIDDEN)
        params["out/bias"] = np.array(f32(HEAD_OUT[b][1], 1)[0])
        weights[head] = params
    return weights


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("engine", help="final_model_all.trt")
    parser.add_argument("--out", required=True, help="where to write the .npz")
    parser.add_argument(
        "--no-hash-check",
        action="store_true",
        help="read an engine whose hash differs (layout unchecked)",
    )
    args = parser.parse_args(argv)
    weights = read_engine(args.engine, check_hash=not args.no_hash_check)
    metadata = dict(
        source=pathlib.Path(args.engine).name,
        engine_sha256=hashlib.sha256(pathlib.Path(args.engine).read_bytes()).hexdigest(),
        note="BatchNorm folded to scale/shift; conv2 and conv3 were FP16 in the engine, the rest FP32",
        heads=list(HEADS),
    )
    # Every weight came out of the engine as FP32 or FP16, so float32 holds them exactly (save_weights checks).
    save_weights(args.out, weights, metadata, dtype=np.float32)
    n = sum(v.size for w in weights.values() for v in w.values())
    print(f"wrote {args.out}: {n} parameters in {len(HEADS)} branches ({', '.join(HEADS)})")


if __name__ == "__main__":
    main()
