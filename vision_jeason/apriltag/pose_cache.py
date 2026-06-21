"""AprilTag 相对位姿缓存。

旧的三坐标调试协议会缓存最近一次有效的 tag1 相对 tag0 位置。
缓存只保存短时间结果，避免 tag 短暂丢失时立即返回无数据。
"""

import time
from typing import Optional, Tuple

Vector3 = Tuple[float, float, float]


class PoseCache:
    """保存一个最近的 XYZ 位姿，并按时间自动过期。"""

    def __init__(self, max_age_s: float = 0.5) -> None:
        self.max_age_s = float(max_age_s)
        self._position_m = None
        self._updated_at_s = None

    def update(self, position_m: Vector3, now_s: Optional[float] = None) -> None:
        """写入最新有效位置。

        ``position_m`` 表示 tag1 原点在 tag0 坐标系下的位置，单位米。
        """
        self._position_m = (
            float(position_m[0]),
            float(position_m[1]),
            float(position_m[2]),
        )
        self._updated_at_s = time.time() if now_s is None else float(now_s)

    def get(self, now_s: Optional[float] = None) -> Optional[Tuple[Vector3, int]]:
        """读取未过期缓存。

        返回 ``(position_m, age_ms)``；若缓存为空或已过期，返回 ``None``。
        """
        if self._position_m is None or self._updated_at_s is None:
            return None
        current_s = time.time() if now_s is None else float(now_s)
        age_s = current_s - self._updated_at_s
        if age_s < 0 or age_s > self.max_age_s:
            return None
        return self._position_m, int(round(age_s * 1000.0))
