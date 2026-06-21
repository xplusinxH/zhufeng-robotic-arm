"""桌面整理目标输出的 ASCII 串口协议。

该协议用于正式桌面整理流程，和 AprilTag 临时标定 JSON 协议分开维护。
坐标输出单位为毫米，帧格式使用 ``@`` 作为帧头、``#`` 作为帧尾，
便于下位机按字节流解析。
"""


def format_no_target() -> str:
    """格式化无目标帧。"""
    return "@NO_TARGET#"


def format_target(class_name: str, x_mm: float, y_mm: float, z_mm: float, score: float) -> str:
    """格式化单目标帧。

    参数：
    - ``class_name``：目标类别，未知物体可传 ``unknown``。
    - ``x_mm/y_mm/z_mm``：工作坐标系或约定输出坐标，单位毫米。
    - ``score``：检测或融合置信度，范围建议为 0 到 1。
    """
    return f"@TARGET,{class_name},{x_mm:.1f},{y_mm:.1f},{z_mm:.1f},{score:.2f}#"
