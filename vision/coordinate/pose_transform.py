"""Small rigid-transform helpers for AprilTag pose calculations."""

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
