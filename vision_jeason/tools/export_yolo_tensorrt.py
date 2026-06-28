"""在 Jetson 本机把 YOLOv8 权重导出为 TensorRT engine。

TensorRT engine 和硬件、CUDA、TensorRT 版本强绑定，不能在 PC 上导出后直接拷到
Jetson 使用。因此本脚本只放在 Jetson 上执行，默认读取当前项目训练出的
``best.pt``，并在同目录生成 ``best.engine``。
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MODEL = PROJECT_ROOT / "models" / "yolov8n_first_best.pt"


def export_tensorrt_engine(
    model_path,
    imgsz=640,
    half=True,
    device=0,
    workspace=None,
):
    """调用 Ultralytics 导出 TensorRT engine，并返回导出文件路径。"""

    from ultralytics import YOLO

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError("找不到 YOLO 权重文件：{0}".format(model_path))

    model = YOLO(str(model_path))
    export_kwargs = {
        "format": "engine",
        "imgsz": int(imgsz),
        "half": bool(half),
        "device": device,
        "simplify": True,
    }
    if workspace is not None:
        export_kwargs["workspace"] = float(workspace)
    exported = model.export(**export_kwargs)
    return Path(exported)


def main(argv=None):
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO .pt 权重路径")
    parser.add_argument("--imgsz", type=int, default=640, help="TensorRT 输入尺寸")
    parser.add_argument("--fp32", action="store_true", help="禁用 FP16，导出 FP32 engine")
    parser.add_argument("--device", default=0, help="Jetson CUDA 设备编号")
    parser.add_argument("--workspace", type=float, help="TensorRT workspace，单位 GiB")
    args = parser.parse_args(argv)

    engine_path = export_tensorrt_engine(
        args.model,
        imgsz=args.imgsz,
        half=not args.fp32,
        device=args.device,
        workspace=args.workspace,
    )
    print("TensorRT engine 已生成：{0}".format(engine_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
