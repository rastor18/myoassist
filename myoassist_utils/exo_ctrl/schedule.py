"""When a fixed-rate controller runs, inside physics that advances one substep at a time."""

from __future__ import annotations

from fractions import Fraction


class TickSchedule:
    """Tick ``k`` of a ``rate_hz`` controller falls on physics substep ``ceil(k * physics_rate_hz / rate_hz)``.

    Counted in whole substeps from ``reset()``, which is itself tick 0, in exact integer arithmetic: the
    ExoBoot's 175 Hz in 1200 Hz physics gives ticks 7, 7, 7, 7, 7, 7 and 6 substeps apart, exactly 175 per
    second on average with no drift, each at most one substep (0.83 ms) after its ideal time ``k / rate_hz``
    and never before it. The physics rate stays what it is; the controller holds its output in between, as
    the boot holds its last motor command.

    A controller faster than the physics would need more than one tick per substep, so that is refused.
    """

    def __init__(self, *, rate_hz: float, physics_rate_hz: float):
        # Fraction of a float is exact (175.0 -> 175), so the ratio below carries no rounding.
        rate, physics_rate = Fraction(rate_hz), Fraction(physics_rate_hz)
        if not rate > 0:
            raise ValueError(f"rate_hz must be > 0, got {rate_hz}")
        if rate > physics_rate:
            raise ValueError(
                f"rate_hz ({rate_hz}) is above the physics rate ({physics_rate_hz}); a tick needs a substep of its own"
            )
        substeps_per_tick = physics_rate / rate
        self.rate_hz = float(rate_hz)
        self.physics_rate_hz = float(physics_rate_hz)
        self._num = substeps_per_tick.numerator
        self._den = substeps_per_tick.denominator
        self.reset()

    def reset(self) -> None:
        self._substep = 0
        self._ticks = 0
        self._next_tick_substep = 0

    def tick(self) -> bool:
        """Advance one physics substep; True if the controller runs on this one. Call exactly once per substep."""
        due = self._substep == self._next_tick_substep
        if due:
            self._ticks += 1
            self._next_tick_substep = -((-self._ticks * self._num) // self._den)  # integer ceil
        self._substep += 1
        return due
