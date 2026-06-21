"""AprilTag 与机械臂临时标定使用的刚体变换工具。

坐标约定：
已有 AprilTag 检测结果为 ``T_camera_tag``，即 tag 坐标系在相机坐标系下的
位姿。若要计算 ``tag_base_ref -> tag_tool0``，使用
``inverse(T_camera_base) * T_camera_tool``。

长度单位保持输入单位不变；当前项目默认使用米，串口旧协议输出时再转毫米。
"""

import math
from typing import Iterable, List, Sequence, Tuple

Matrix4 = List[List[float]]
Vector3 = Tuple[float, float, float]


def make_transform(rotation: Sequence[Sequence[float]], translation: Iterable[float]) -> Matrix4:
    """由 3x3 旋转矩阵和 XYZ 平移构造 4x4 齐次变换矩阵。"""
    tx, ty, tz = [float(value) for value in translation]
    return [
        [float(rotation[0][0]), float(rotation[0][1]), float(rotation[0][2]), tx],
        [float(rotation[1][0]), float(rotation[1][1]), float(rotation[1][2]), ty],
        [float(rotation[2][0]), float(rotation[2][1]), float(rotation[2][2]), tz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def invert_transform(transform: Sequence[Sequence[float]]) -> Matrix4:
    """求刚体 4x4 变换矩阵的逆矩阵。"""
    rotation_t = [
        [float(transform[0][0]), float(transform[1][0]), float(transform[2][0])],
        [float(transform[0][1]), float(transform[1][1]), float(transform[2][1])],
        [float(transform[0][2]), float(transform[1][2]), float(transform[2][2])],
    ]
    translation = [
        float(transform[0][3]),
        float(transform[1][3]),
        float(transform[2][3]),
    ]
    inverse_translation = [
        -sum(rotation_t[row][col] * translation[col] for col in range(3))
        for row in range(3)
    ]
    return make_transform(rotation_t, inverse_translation)


def multiply_transform(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix4:
    """矩阵乘法：返回 ``left * right``。"""
    return [
        [
            sum(float(left[row][index]) * float(right[index][col]) for index in range(4))
            for col in range(4)
        ]
        for row in range(4)
    ]


def relative_transform(
    reference_in_camera: Sequence[Sequence[float]],
    target_in_camera: Sequence[Sequence[float]],
) -> Matrix4:
    """将目标 tag 位姿转换到参考 tag 坐标系下。"""
    return multiply_transform(invert_transform(reference_in_camera), target_in_camera)


def transform_translation(transform: Sequence[Sequence[float]]) -> Vector3:
    """从 4x4 变换矩阵中提取 XYZ 平移，单位沿用输入矩阵。"""
    return (
        float(transform[0][3]),
        float(transform[1][3]),
        float(transform[2][3]),
    )


def transform_translation_mm(transform: Sequence[Sequence[float]]) -> Vector3:
    """从 4x4 变换矩阵中提取 XYZ 平移并转换为毫米。"""
    x_m, y_m, z_m = transform_translation(transform)
    return (round(x_m * 1000.0, 6), round(y_m * 1000.0, 6), round(z_m * 1000.0, 6))


def transform_rotation(transform: Sequence[Sequence[float]]) -> List[List[float]]:
    """从 4x4 变换矩阵中提取 3x3 旋转矩阵。"""
    return [
        [float(transform[0][0]), float(transform[0][1]), float(transform[0][2])],
        [float(transform[1][0]), float(transform[1][1]), float(transform[1][2])],
        [float(transform[2][0]), float(transform[2][1]), float(transform[2][2])],
    ]


def rotation_matrix_to_quaternion_xyzw(rotation: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """将 3x3 旋转矩阵转换为 ``x, y, z, w`` 顺序的四元数。"""
    r00 = float(rotation[0][0])
    r01 = float(rotation[0][1])
    r02 = float(rotation[0][2])
    r10 = float(rotation[1][0])
    r11 = float(rotation[1][1])
    r12 = float(rotation[1][2])
    r20 = float(rotation[2][0])
    r21 = float(rotation[2][1])
    r22 = float(rotation[2][2])
    trace = r00 + r11 + r22

    # 按矩阵迹和最大对角元素分支，避免接近 180 度旋转时数值不稳定。
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (r21 - r12) / scale
        qy = (r02 - r20) / scale
        qz = (r10 - r01) / scale
    elif r00 > r11 and r00 > r22:
        scale = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / scale
        qx = 0.25 * scale
        qy = (r01 + r10) / scale
        qz = (r02 + r20) / scale
    elif r11 > r22:
        scale = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / scale
        qx = (r01 + r10) / scale
        qy = 0.25 * scale
        qz = (r12 + r21) / scale
    else:
        scale = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / scale
        qx = (r02 + r20) / scale
        qy = (r12 + r21) / scale
        qz = 0.25 * scale

    return normalize_quaternion_xyzw((qx, qy, qz, qw))


def normalize_quaternion_xyzw(quaternion: Sequence[float]) -> Tuple[float, float, float, float]:
    """归一化 ``x, y, z, w`` 顺序的四元数。"""
    qx, qy, qz, qw = [float(value) for value in quaternion]
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (qx / norm, qy / norm, qz / norm, qw / norm)


def average_quaternions_xyzw(quaternions: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """对多个四元数做平均。

    四元数 ``q`` 和 ``-q`` 表示同一个姿态。平均前先统一到同一半球，
    避免方向相同但符号相反的样本相互抵消。
    """
    if not quaternions:
        return (0.0, 0.0, 0.0, 1.0)
    reference = normalize_quaternion_xyzw(quaternions[0])
    sums = [0.0, 0.0, 0.0, 0.0]
    for quaternion in quaternions:
        current = normalize_quaternion_xyzw(quaternion)
        dot = sum(reference[index] * current[index] for index in range(4))
        if dot < 0.0:
            current = tuple(-value for value in current)
        for index in range(4):
            sums[index] += current[index]
    return normalize_quaternion_xyzw(sums)


def quaternion_xyzw_to_rotation_matrix(quaternion: Sequence[float]) -> List[List[float]]:
    """将 ``x, y, z, w`` 顺序四元数转换为 3x3 旋转矩阵。"""
    x, y, z, w = normalize_quaternion_xyzw(quaternion)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def transform_pose_xyzw(transform: Sequence[Sequence[float]]) -> Tuple[Vector3, Tuple[float, float, float, float]]:
    """从 4x4 变换矩阵中提取位置和 XYZW 四元数姿态。"""
    return transform_translation(transform), rotation_matrix_to_quaternion_xyzw(transform_rotation(transform))
