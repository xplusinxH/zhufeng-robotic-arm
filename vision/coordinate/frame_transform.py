"""eye-in-hand 坐标链工具。

本模块处理机械臂基座、末端工具和相机之间的刚体变换：

``T_base_camera = T_base_tool * T_tool_camera``

其中 ``T_base_tool`` 由机械臂控制端通过串口实时发送，
``T_tool_camera`` 来自后续手眼标定且在相机安装不变时固定。
"""

from typing import Iterable, Sequence, Tuple

from coordinate.pose_transform import (
    make_transform,
    multiply_transform,
    quaternion_xyzw_to_rotation_matrix,
)

Point3 = Tuple[float, float, float]


def make_transform_from_pose_xyzw(
    x_m: float,
    y_m: float,
    z_m: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
):
    """由平移和 XYZW 四元数构造 4x4 齐次变换矩阵。

    平移单位为米；四元数表示父坐标系到子坐标系的姿态。
    """
    rotation = quaternion_xyzw_to_rotation_matrix((qx, qy, qz, qw))
    return make_transform(rotation, (x_m, y_m, z_m))


def compose_transform(parent_from_middle, middle_from_child):
    """组合两段坐标变换，返回 ``parent_from_child``。"""
    return multiply_transform(parent_from_middle, middle_from_child)


def transform_point(transform: Sequence[Sequence[float]], point: Iterable[float]) -> Point3:
    """将三维点从子坐标系转换到父坐标系。

    ``transform`` 为 4x4 齐次变换矩阵，``point`` 单位沿用输入数据；
    当前项目中默认使用米。
    """
    x_m, y_m, z_m = [float(value) for value in point]
    return (
        float(transform[0][0]) * x_m
        + float(transform[0][1]) * y_m
        + float(transform[0][2]) * z_m
        + float(transform[0][3]),
        float(transform[1][0]) * x_m
        + float(transform[1][1]) * y_m
        + float(transform[1][2]) * z_m
        + float(transform[1][3]),
        float(transform[2][0]) * x_m
        + float(transform[2][1]) * y_m
        + float(transform[2][2]) * z_m
        + float(transform[2][3]),
    )
