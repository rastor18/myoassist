"""Ports of the ExoBoot's real-time filter and delay timer, on a passed-in clock.

The boot steps these once per main-loop iteration, at 175 Hz (``TARGET_FREQ``), and its filters are designed for that
rate: each call is one sample. So they must be stepped once per controller tick here too, never per physics substep.
Plain Python floats throughout, since they run inside the physics loop.
"""

from __future__ import annotations

from scipy import signal


class Butterworth:
    """Port of ExoBoot ``filters.Butterworth`` (filters.py:22-51): a causal Butterworth as second-order sections.

    Same design: ``scipy.signal.butter(N, Wn=cutoff / (fs / 2), output='sos')``. Same start: the state is
    ``sosfilt_zi`` scaled by the first sample, so a constant input passes through unchanged from the first call. Each
    call then runs one sample through scipy's own ``sosfilt`` recurrence (transposed direct form II, per section), with
    the same order of operations, so the output is what the boot's ``signal.sosfilt(sos, [x], zi=zi)`` returns.
    """

    def __init__(self, *, order: int, cutoff_hz: float, fs_hz: float, btype: str = "low"):
        if not 0 < cutoff_hz < fs_hz / 2:
            raise ValueError(f"need 0 < cutoff_hz < fs_hz / 2, got cutoff {cutoff_hz} Hz at fs {fs_hz} Hz")
        sos = signal.butter(N=order, Wn=cutoff_hz / (fs_hz / 2), btype=btype, output="sos")
        # (b0, b1, b2, a1, a2) per section; scipy normalizes a0 to 1.
        self._sections = [(float(s[0]), float(s[1]), float(s[2]), float(s[4]), float(s[5])) for s in sos]
        self._zi = [(float(z[0]), float(z[1])) for z in signal.sosfilt_zi(sos)]
        self.order = order
        self.cutoff_hz = cutoff_hz
        self.fs_hz = fs_hz
        self.reset()

    def reset(self) -> None:
        self._state: list[list[float]] | None = None  # seeded from the first sample

    def filter(self, x: float) -> float:
        x = float(x)
        state = self._state
        if state is None:
            state = self._state = [[z0 * x, z1 * x] for z0, z1 in self._zi]
        for (b0, b1, b2, a1, a2), z in zip(self._sections, state):
            y = b0 * x + z[0]
            z[0] = b1 * x - a1 * y + z[1]
            z[1] = b2 * x - a2 * y
            x = y
        return x


class DelayTimer:
    """Port of ExoBoot ``util.DelayTimer`` (its default mode) on a passed-in clock rather than ``time.perf_counter()``.

    ``check`` is True once strictly more than ``delay_time`` has passed since ``start``, and stays True until ``reset``.
    ``start`` while running restarts it.
    """

    def __init__(self, delay_time: float):
        if not delay_time >= 0:
            raise ValueError(f"delay_time must be >= 0, got {delay_time}")
        self.delay_time = delay_time
        self.reset()

    def start(self, t: float) -> None:
        self.start_time = t

    def check(self, t: float) -> bool:
        return self.start_time is not None and t > self.start_time + self.delay_time

    def reset(self) -> None:
        self.start_time: float | None = None
