"""计算并缓存 tag1 在 tag0 坐标系下的位置。

这是早期 ``@GET_TOOL#`` 三坐标调试协议使用的服务层。新的标定串口协议
使用完整 6D JSON，但保留此模块便于历史工具和单元测试继续工作。
"""

from typing import Dict, Optional, Sequence, Tuple

from apriltag.pose_cache import PoseCache
from communication.tag_pose_protocol import format_no_tag, format_tag_pose
from coordinate.pose_transform import relative_transform, transform_translation

Matrix4 = Sequence[Sequence[float]]
Vector3 = Tuple[float, float, float]


class TagPoseService:
    """维护末端 tag 相对底座 tag 的最新位置。"""

    def __init__(self, base_tag_id: int = 0, tool_tag_id: int = 1, max_age_s: float = 0.5) -> None:
        self.base_tag_id = int(base_tag_id)
        self.tool_tag_id = int(tool_tag_id)
        self.cache = PoseCache(max_age_s=max_age_s)

    def update_from_detections(
        self, detections: Dict[int, Matrix4], now_s: Optional[float] = None
    ) -> bool:
        """从 ``camera -> tag`` 检测结果更新缓存。

        只有同时看到底座 tag 和末端 tag 时才更新；否则保持旧缓存不变。
        """
        if self.base_tag_id not in detections or self.tool_tag_id not in detections:
            return False

        base_to_tool = relative_transform(
            detections[self.base_tag_id],
            detections[self.tool_tag_id],
        )
        self.cache.update(transform_translation(base_to_tool), now_s=now_s)
        return True

    def get_cached(self, now_s: Optional[float] = None) -> Optional[Tuple[Vector3, int]]:
        """返回最新未过期缓存位置和年龄。"""
        return self.cache.get(now_s=now_s)

    def format_response(self, now_s: Optional[float] = None) -> str:
        """按旧串口协议格式化当前缓存结果。"""
        cached = self.get_cached(now_s=now_s)
        if cached is None:
            return format_no_tag()
        position_m, age_ms = cached
        return format_tag_pose(position_m, age_ms)
