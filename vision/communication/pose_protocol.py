"""机械臂实时末端位姿串口输入协议。

控制端向 Jetson 发送当前 ``T_base_tool``：

``@POSE,x,y,z,qx,qy,qz,qw#``

单位约定：
- ``x/y/z``：米，表示 tool 原点在 base 坐标系下的位置。
- ``qx/qy/qz/qw``：XYZW 四元数，表示 tool 坐标系在 base 坐标系下的姿态。

Jetson 收到后结合手眼标定 ``T_tool_camera``，得到实时 ``T_base_camera``。
"""

from coordinate.frame_transform import make_transform_from_pose_xyzw

POSE_COMMAND_PREFIX = "@POSE,"


def is_pose_frame(message: str) -> bool:
    """判断串口帧是否为实时末端位姿帧。"""
    text = message.strip()
    return text.startswith(POSE_COMMAND_PREFIX) and text.endswith("#")


def parse_pose_frame(message: str):
    """解析 ``@POSE`` 帧并返回位姿字段和 4x4 变换矩阵。

    返回字典包含：
    - ``translation_m``：``(x, y, z)``，单位米。
    - ``orientation_xyzw``：四元数。
    - ``transform``：``T_base_tool``。
    """
    text = message.strip()
    if not is_pose_frame(text):
        raise ValueError("不是有效的 @POSE 位姿帧")
    fields = text[len(POSE_COMMAND_PREFIX) : -1].split(",")
    if len(fields) != 7:
        raise ValueError("@POSE 位姿帧必须包含 7 个数值")
    values = [float(field) for field in fields]
    x_m, y_m, z_m, qx, qy, qz, qw = values
    return {
        "translation_m": (x_m, y_m, z_m),
        "orientation_xyzw": (qx, qy, qz, qw),
        "transform": make_transform_from_pose_xyzw(x_m, y_m, z_m, qx, qy, qz, qw),
    }
