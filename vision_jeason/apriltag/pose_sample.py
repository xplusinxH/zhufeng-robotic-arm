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


def _median(values):
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0
