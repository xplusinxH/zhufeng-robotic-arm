"""Cache the most recent valid AprilTag relative pose."""

import time
from typing import Optional, Tuple

Vector3 = Tuple[float, float, float]


class PoseCache:
    """Store one recent XYZ pose and expire it after a configurable age."""

    def __init__(self, max_age_s: float = 0.5) -> None:
        self.max_age_s = float(max_age_s)
        self._position_m = None
        self._updated_at_s = None

    def update(self, position_m: Vector3, now_s: Optional[float] = None) -> None:
        """Store the latest valid tag1-in-tag0 position in meters."""
        self._position_m = (
            float(position_m[0]),
            float(position_m[1]),
            float(position_m[2]),
        )
        self._updated_at_s = time.time() if now_s is None else float(now_s)

    def get(self, now_s: Optional[float] = None) -> Optional[Tuple[Vector3, int]]:
        """Return ``(position_m, age_ms)`` when the cached pose is still fresh."""
        if self._position_m is None or self._updated_at_s is None:
            return None
        current_s = time.time() if now_s is None else float(now_s)
        age_s = current_s - self._updated_at_s
        if age_s < 0 or age_s > self.max_age_s:
            return None
        return self._position_m, int(round(age_s * 1000.0))
