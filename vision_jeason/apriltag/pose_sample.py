"""Robust fusion helpers for multiple AprilTag pose frames."""

from typing import Sequence

from coordinate.pose_transform import (
    average_quaternions_xyzw,
    make_transform,
    quaternion_xyzw_to_rotation_matrix,
    rotation_matrix_to_quaternion_xyzw,
    transform_rotation,
    transform_translation,
)


def robust_average_transforms(transforms: Sequence[Sequence[Sequence[float]]]):
    """Fuse transforms using median XYZ and averaged quaternion orientation."""
    if not transforms:
        raise ValueError("至少需要一个有效位姿用于融合")

    translations = [transform_translation(transform) for transform in transforms]
    quaternions = [
        rotation_matrix_to_quaternion_xyzw(transform_rotation(transform))
        for transform in transforms
    ]
    translation = (
        _median([item[0] for item in translations]),
        _median([item[1] for item in translations]),
        _median([item[2] for item in translations]),
    )
    quaternion = average_quaternions_xyzw(quaternions)
    return make_transform(quaternion_xyzw_to_rotation_matrix(quaternion), translation)


class BaseReferenceCache:
    """Keep recent camera-to-base-ref observations and return their fused pose."""

    def __init__(self, max_items=20):
        self.max_items = int(max_items)
        self._items = []

    def add(self, camera_to_base_ref):
        """Add one camera-to-base-ref transform."""
        self._items.append(camera_to_base_ref)
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items :]

    def has_value(self):
        """Return whether at least one base reference observation is cached."""
        return bool(self._items)

    def get_fused(self):
        """Return the fused camera-to-base-ref transform, or None if empty."""
        if not self._items:
            return None
        return robust_average_transforms(self._items)


def _median(values):
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0
