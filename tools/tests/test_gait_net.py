"""The 4-headed gait network's NumPy passes, and the tool that recovers its weights from the Jetson's engine.

The real weights come from the engine (``tools/recover_gait_net.py``), which is not in the repo, so most of these use
random weights of the same architecture: what matters here is that the streaming pass is the full-window pass on the
Jetson's zero-filled shift buffer. The recovered network itself was checked against the DL validation sessions' logs with
``tools/replay_dl_session.py``. Set MYOASSIST_GAIT_NET_ENGINE to the engine to also run the recovery test.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from myoassist_utils.exo_ctrl.gait_net import (
    HEADS,
    PARAM_SHAPES,
    WINDOW,
    FullWindowGaitNet,
    StreamingGaitNet,
    load_weights,
    save_weights,
    validate_weights,
)

# Rough scales of the channels as the boot sends them: accel in g, gyro in deg/s, ankle angle in deg, its velocity in deg/s.
CHANNEL_SCALE = np.array([1.0, 1.0, 0.3, 50.0, 50.0, 150.0, 10.0, 100.0])


def random_weights(seed: int = 0):
    rng = np.random.default_rng(seed)
    weights = {}
    for head in HEADS:
        params = {}
        for name, shape in PARAM_SHAPES.items():
            fan_in = int(np.prod(shape[1:])) if name.endswith("kernel") and len(shape) > 1 else max(1, shape[0] if shape else 1)
            params[name] = rng.normal(0.0, 1.0 / np.sqrt(fan_in), shape)
        params["conv1/kernel"] /= CHANNEL_SCALE[None, :, None]  # the network normalizes raw units itself
        params["bn1/scale"] = rng.uniform(0.5, 1.5, PARAM_SHAPES["bn1/scale"])
        params["bn2/scale"] = rng.uniform(0.5, 1.5, PARAM_SHAPES["bn2/scale"])
        weights[head] = params
    return weights


def walking_samples(n: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 175.0
    phases = rng.uniform(0, 2 * np.pi, 8)
    return CHANNEL_SCALE * (np.sin(2 * np.pi * t[:, None] / 1.1 + phases) + 0.1 * rng.normal(size=(n, 8)))


def buffer_windows(samples: np.ndarray) -> np.ndarray:
    """The Jetson's buffer after each sample: zero-filled, newest last."""
    padded = np.vstack([np.zeros((WINDOW, samples.shape[1])), samples])
    return np.stack([padded[i + 1 : i + 1 + WINDOW] for i in range(len(samples))])


@pytest.fixture(scope="module")
def weights():
    return random_weights()


def test_streaming_is_the_full_window_on_a_zero_filled_buffer(weights):
    """Through the zero-filled start and well past one full window, every head, to rounding."""
    samples = walking_samples(320)
    want = FullWindowGaitNet(weights)(buffer_windows(samples))
    net = StreamingGaitNet(weights)
    got = [net.step(s[None]) for s in samples]
    for head in HEADS:
        np.testing.assert_allclose([g[head][0] for g in got], want[head], rtol=0, atol=1e-10, err_msg=head)


def test_streams_are_independent_and_reset_separately(weights):
    a, b = walking_samples(250, seed=2), walking_samples(250, seed=3)
    both = StreamingGaitNet(weights, n_streams=2)
    for sa, sb in zip(a, b):
        out = both.step(np.stack([sa, sb]))
    for stream, samples in enumerate((a, b)):
        alone = StreamingGaitNet(weights)
        for s in samples:
            ref = alone.step(s[None])
        for head in HEADS:
            assert out[head][stream] == pytest.approx(ref[head][0], abs=1e-12), (
                head,
                stream,
            )
    # Resetting one stream restarts it alone.
    both.reset(streams=[1])
    fresh = StreamingGaitNet(weights)
    sample = walking_samples(1, seed=4)[0]
    out, ref = both.step(np.stack([sample, sample])), fresh.step(sample[None])
    for head in HEADS:
        assert out[head][1] == pytest.approx(ref[head][0], abs=1e-12)
        assert out[head][0] != pytest.approx(ref[head][0], abs=1e-9), "stream 0 kept its history"


def test_a_subset_of_heads_gives_the_same_values(weights):
    samples = walking_samples(220)
    full = FullWindowGaitNet(weights)(buffer_windows(samples))
    sub = StreamingGaitNet(weights, heads=("stance_swing", "stance_phase"))
    for s in samples:
        got = sub.step(s[None])
    assert set(got) == {"stance_swing", "stance_phase"}
    for head in got:
        assert got[head][0] == pytest.approx(full[head][-1], abs=1e-10)


def test_float32_stays_close(weights):
    samples = walking_samples(260)
    f64, f32 = StreamingGaitNet(weights), StreamingGaitNet(weights, dtype=np.float32)
    for s in samples:
        a, b = f64.step(s[None]), f32.step(s[None])
    for head in HEADS:
        assert float(b[head][0]) == pytest.approx(float(a[head][0]), rel=1e-3, abs=1e-4), head


def test_snapshot_and_restore(weights):
    samples = walking_samples(60)
    net = StreamingGaitNet(weights)
    for s in samples[:30]:
        net.step(s[None])
    saved = net.snapshot()
    first = [net.step(s[None]) for s in samples[30:]]
    net.restore(saved)
    again = [net.step(s[None]) for s in samples[30:]]
    for x, y in zip(first, again):
        for head in HEADS:
            assert x[head][0] == y[head][0]


def test_stance_swing_is_a_probability(weights):
    out = FullWindowGaitNet(weights)(buffer_windows(walking_samples(240)))
    assert np.all((out["stance_swing"] > 0) & (out["stance_swing"] < 1))


def test_weights_round_trip_through_npz(weights, tmp_path):
    path = tmp_path / "net.npz"
    save_weights(path, weights, metadata={"source": "test"})
    loaded, metadata = load_weights(path)
    assert metadata == {"source": "test"}
    for head in HEADS:
        for name in PARAM_SHAPES:
            np.testing.assert_array_equal(loaded[head][name], weights[head][name])


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda w: w.pop("ramp"), "need weights for heads"),
        (lambda w: w["velocity"].pop("conv4/bias"), "missing"),
        (
            lambda w: w["velocity"].__setitem__("dense1/kernel", np.zeros((23, 43))),
            "shape",
        ),
        (lambda w: w["stance_phase"]["bn2/shift"].__setitem__(0, np.nan), "not finite"),
    ],
)
def test_bad_weights_are_refused(corrupt, message):
    w = random_weights()
    corrupt(w)
    with pytest.raises(ValueError, match=message):
        validate_weights(w)


def test_the_recovery_tool_refuses_another_engine(tmp_path):
    from tools.recover_gait_net import read_engine

    path = tmp_path / "other.trt"
    path.write_bytes(b"ptrt" + bytes(1000))
    with pytest.raises(ValueError, match="sha256"):
        read_engine(path)


@pytest.mark.skipif(
    not os.environ.get("MYOASSIST_GAIT_NET_ENGINE"),
    reason="set MYOASSIST_GAIT_NET_ENGINE to the .trt",
)
def test_recovers_the_engine():
    from tools.recover_gait_net import read_engine

    weights = read_engine(os.environ["MYOASSIST_GAIT_NET_ENGINE"])
    validate_weights(weights)
    assert sum(v.size for w in weights.values() for v in w.values()) == 720492
