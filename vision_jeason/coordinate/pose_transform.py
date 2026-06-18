"""Small rigid-transform helpers for AprilTag pose calculations."""

import math
from typing import Iterable, List, Sequence, Tuple

Matrix4 = List[List[float]]
Vector3 = Tuple[float, float, float]


def make_transform(rotation: Sequence[Sequence[float]], translation: Iterable[float]) -> Matrix4:
    """Build a 4x4 rigid transform from a 3x3 rotation and XYZ translation."""
    tx, ty, tz = [float(value) for value in translation]
    return [
        [float(rotation[0][0]), float(rotation[0][1]), float(rotation[0][2]), tx],
        [float(rotation[1][0]), float(rotation[1][1]), float(rotation[1][2]), ty],
        [float(rotation[2][0]), float(rotation[2][1]), float(rotation[2][2]), tz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def invert_transform(transform: Sequence[Sequence[float]]) -> Matrix4:
    """Invert a rigid 4x4 transform."""
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
    """Multiply two 4x4 transforms."""
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
    """Return target pose expressed in the reference tag frame."""
    return multiply_transform(invert_transform(reference_in_camera), target_in_camera)


def transform_translation(transform: Sequence[Sequence[float]]) -> Vector3:
    """Extract XYZ translation from a 4x4 transform in meters."""
    return (
        float(transform[0][3]),
        float(transform[1][3]),
        float(transform[2][3]),
    )


def transform_translation_mm(transform: Sequence[Sequence[float]]) -> Vector3:
    """Extract XYZ translation from a 4x4 transform in millimeters."""
    x_m, y_m, z_m = transform_translation(transform)
    return (round(x_m * 1000.0, 6), round(y_m * 1000.0, 6), round(z_m * 1000.0, 6))


def transform_rotation(transform: Sequence[Sequence[float]]) -> List[List[float]]:
    """Extract a 3x3 rotation matrix from a 4x4 transform."""
    return [
        [float(transform[0][0]), float(transform[0][1]), float(transform[0][2])],
        [float(transform[1][0]), float(transform[1][1]), float(transform[1][2])],
        [float(transform[2][0]), float(transform[2][1]), float(transform[2][2])],
    ]


def rotation_matrix_to_quaternion_xyzw(rotation: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to an ``x, y, z, w`` quaternion."""
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
    """Normalize an ``x, y, z, w`` quaternion."""
    qx, qy, qz, qw = [float(value) for value in quaternion]
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (qx / norm, qy / norm, qz / norm, qw / norm)


def average_quaternions_xyzw(quaternions: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """Average quaternions after aligning them to the same hemisphere."""
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
    """Convert an ``x, y, z, w`` quaternion to a 3x3 rotation matrix."""
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
    """Extract XYZ position and XYZW quaternion from a 4x4 transform."""
    return transform_translation(transform), rotation_matrix_to_quaternion_xyzw(transform_rotation(transform))
