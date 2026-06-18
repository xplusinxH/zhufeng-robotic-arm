"""Compute and serve tag1-in-tag0 positions."""

from typing import Dict, Optional, Sequence, Tuple

from apriltag.pose_cache import PoseCache
from communication.tag_pose_protocol import format_no_tag, format_tag_pose
from coordinate.pose_transform import relative_transform, transform_translation

Matrix4 = Sequence[Sequence[float]]
Vector3 = Tuple[float, float, float]


class TagPoseService:
    """Maintain the latest end-tag position relative to the base tag."""

    def __init__(self, base_tag_id: int = 1, tool_tag_id: int = 0, max_age_s: float = 0.5) -> None:
        self.base_tag_id = int(base_tag_id)
        self.tool_tag_id = int(tool_tag_id)
        self.cache = PoseCache(max_age_s=max_age_s)

    def update_from_detections(
        self, detections: Dict[int, Matrix4], now_s: Optional[float] = None
    ) -> bool:
        """Update cache from detected camera-to-tag transforms."""
        if self.base_tag_id not in detections or self.tool_tag_id not in detections:
            return False

        base_to_tool = relative_transform(
            detections[self.base_tag_id],
            detections[self.tool_tag_id],
        )
        self.cache.update(transform_translation(base_to_tool), now_s=now_s)
        return True

    def get_cached(self, now_s: Optional[float] = None) -> Optional[Tuple[Vector3, int]]:
        """Return the latest fresh cached position and age."""
        return self.cache.get(now_s=now_s)

    def format_response(self, now_s: Optional[float] = None) -> str:
        """Format the current serial response."""
        cached = self.get_cached(now_s=now_s)
        if cached is None:
            return format_no_tag()
        position_m, age_ms = cached
        return format_tag_pose(position_m, age_ms)
