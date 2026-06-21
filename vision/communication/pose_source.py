"""末端位姿输入源。

真实控制侧串口尚未就绪时，本模块允许 Jetson/PC 从文本文件读取模拟
``T_base_tool``。后续接入真实串口时，主循环可以继续消费同样的数据结构，
只替换位姿来源即可。
"""

from pathlib import Path

from communication.pose_protocol import is_pose_frame, parse_pose_frame


def load_base_tool_pose_from_file(input_path):
    """从手动文件中读取一帧 ``T_base_tool``。

    当前支持最小、最稳的格式：文件中第一条非空非注释行写入控制侧未来要
    发送的 ``@POSE,x,y,z,qx,qy,qz,qw#`` 帧。这样离线调试文件和后续串口帧
    使用同一套解析逻辑，不会出现两套协议互相漂移。
    """

    input_path = Path(input_path)
    for raw_line in input_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not is_pose_frame(line):
            raise ValueError("模拟末端位姿文件必须包含 @POSE,...# 帧")
        return parse_pose_frame(line)
    raise ValueError("模拟末端位姿文件中没有有效位姿帧")
