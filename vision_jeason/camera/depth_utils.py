"""深度值有效性判断工具。

当前项目优先处理桌面近距离物体，默认有效范围为 0.15 m 到 1.20 m。
后续如相机安装高度变化，应从配置文件传入新的范围。
"""


def is_valid_depth(depth_m: float, min_m: float = 0.15, max_m: float = 1.20) -> bool:
    """判断深度值是否位于工作范围内，单位米。"""
    return min_m <= depth_m <= max_m
