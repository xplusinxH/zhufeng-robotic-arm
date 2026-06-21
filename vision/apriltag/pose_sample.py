"""AprilTag 多帧位姿融合工具。

单帧 AprilTag 位姿会有抖动，因此串口响应前会连续采样多帧：
平移取逐轴中位数，姿态取四元数平均。输入和输出均为 4x4 齐次变换矩阵。
"""

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
    """融合多帧刚体变换。

    平移使用中位数，能抑制少量异常帧；姿态使用半球对齐后的四元数平均。
    """
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
    """缓存最近的 ``camera -> base_ref`` 观测并返回融合结果。"""

    def __init__(self, max_items=20):
        self.max_items = int(max_items)
        self._items = []

    def add(self, camera_to_base_ref):
        """加入一帧 ``camera -> base_ref`` 变换。"""
        self._items.append(camera_to_base_ref)
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items :]

    def has_value(self):
        """判断是否已有可用底座参考 tag 缓存。"""
        return bool(self._items)

    def get_fused(self):
        """返回融合后的 ``camera -> base_ref`` 变换；缓存为空时返回 ``None``。"""
        if not self._items:
            return None
        return robust_average_transforms(self._items)


def _median(values):
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0
