from __future__ import annotations

import math
from dataclasses import dataclass

from control.safety import ServoLimits, SafetyError


@dataclass(frozen=True)
class TailGait:
    center_angle: float = 90.0
    amplitude_deg: float = 5.0
    frequency_hz: float = 0.5

    def angle_at(self, seconds: float, limits: ServoLimits) -> float:
        if self.amplitude_deg > 5:
            raise SafetyError("Default gait generation is limited to 5 degrees during bring-up.")

        angle = self.center_angle + self.amplitude_deg * math.sin(2 * math.pi * self.frequency_hz * seconds)
        return limits.validate(angle)

