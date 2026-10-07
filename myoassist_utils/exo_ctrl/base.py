"""Interfaces for scripted exoskeleton controllers.

The decomposition mirrors the ExoBoot's own ``gait_state_estimators.GaitStateEstimator``, which composes
a swappable heel-strike detector with a gait-phase estimator that does not care where heel strikes come
from. On hardware the detector reads a shank gyro. The MyoAssist models have no gyro sensor, so in
simulation the detector reads foot GRF instead. Keeping the seam at the detector -- not at a coarser
"phase source" -- is what lets a simulated-gyro detector drop in later without touching the phase
estimator or the torque profile.
"""

from __future__ import annotations

from typing import Protocol


class HeelStrikeDetector(Protocol):
    """Turns one leg's sensor signal into heel-strike events."""

    # When the last detected strike happened. A detector that confirms a strike only after the fact dates it back to
    # when it began; for the others it is the time of the tick that detected it.
    strike_time: float

    def reset(self) -> None: ...

    def detect(self, t: float, signal: float) -> bool:
        """Return True on the tick a heel strike is detected. ``t`` is simulation time in seconds."""
        ...


class LegExoController(Protocol):
    """One leg's scripted controller: a sensor signal in, plantarflexion torque in N*m out.

    Torque is plantarflexion-positive, matching the ExoBoot. The in-loop seam turns it into the joint's sign
    (``device.FixedRateLegExos``) and then into ctrl (``rl_train.envs.device_control.JointTorqueDrive``).
    """

    def reset(self) -> None: ...

    def step(self, t: float, signal: float) -> float: ...

    def diagnostics(self) -> dict[str, float]:
        """Scalars describing the last step. Always finite: these may be summed into training logs."""
        ...
