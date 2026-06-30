from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    output_min: float | None = None
    output_max: float | None = None

    integral: float = 0.0
    previous_error: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        if dt <= 0:
            raise ValueError("dt must be positive")

        error = setpoint - measurement
        self.integral += error * dt
        derivative = 0.0 if self.previous_error is None else (error - self.previous_error) / dt
        self.previous_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        if self.output_min is not None:
            output = max(self.output_min, output)
        if self.output_max is not None:
            output = min(self.output_max, output)
        return output

