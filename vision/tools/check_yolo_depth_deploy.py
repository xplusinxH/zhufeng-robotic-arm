"""Jetson YOLO 深度识别部署自检。

在导出 TensorRT 或启动串口服务前，先运行本脚本确认模型文件、Python 依赖、
RealSense SDK 和项目入口是否可用。它不打开相机流，避免自检阶段占用设备。
"""

import argparse
import importlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MODEL = PROJECT_ROOT / "models" / "yolov8n_first_best.pt"
DEFAULT_ENGINE = PROJECT_ROOT / "models" / "yolov8n_first_best.engine"


def make_check_item(name, status, message):
    """构造稳定的自检项结构。"""

    return {
        "name": str(name),
        "status": str(status),
        "message": str(message),
    }


def build_deploy_check_summary(items):
    """汇总自检结果。

    ``error`` 表示不能继续部署；``warn`` 表示可以继续，但通常需要先导出 engine
    或注意性能退化。
    """

    error_count = sum(1 for item in items if item["status"] == "error")
    warn_count = sum(1 for item in items if item["status"] == "warn")
    if error_count:
        status = "error"
    elif warn_count:
        status = "warn"
    else:
        status = "ok"
    return {
        "ready": error_count == 0,
        "status": status,
        "error_count": error_count,
        "warn_count": warn_count,
        "items": list(items),
    }


def run_deploy_checks(model_path=DEFAULT_MODEL, engine_path=DEFAULT_ENGINE):
    """执行 Jetson 部署前自检。"""

    model_path = Path(model_path)
    engine_path = Path(engine_path)
    items = [
        _check_file("model_pt", model_path, required=True),
        _check_file("model_engine", engine_path, required=False),
        _check_import("numpy", "numpy"),
        _check_import("ultralytics", "ultralytics"),
        _check_import("serial", "pyserial"),
        _check_import("pyrealsense2", "pyrealsense2"),
        _check_import("perception.yolo_depth_geometry", "项目 YOLO 深度几何模块"),
        _check_import("tools.run_yolo_depth_scene", "项目单帧运行入口"),
        _check_import("tools.serve_yolo_depth_serial", "项目串口服务入口"),
    ]
    return build_deploy_check_summary(items)


def _check_file(name, path, required):
    """检查文件是否存在。"""

    if Path(path).exists():
        return make_check_item(name, "ok", "存在：{0}".format(path))
    if required:
        return make_check_item(name, "error", "缺少必要文件：{0}".format(path))
    return make_check_item(name, "warn", "可选文件不存在：{0}".format(path))


def _check_import(module_name, display_name, required=True):
    """检查 Python 模块是否可导入。"""

    try:
        importlib.import_module(module_name)
    except Exception as exc:
        status = "error" if required else "warn"
        return make_check_item(display_name, status, "导入失败：{0}".format(exc))
    return make_check_item(display_name, "ok", "导入成功")


def main(argv=None):
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO .pt 权重路径")
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE, help="YOLO TensorRT engine 路径")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON")
    args = parser.parse_args(argv)

    summary = run_deploy_checks(model_path=args.model, engine_path=args.engine)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        for item in summary["items"]:
            print("[{0}] {1}: {2}".format(item["status"], item["name"], item["message"]))
        print(
            "summary: status={0} ready={1} errors={2} warnings={3}".format(
                summary["status"],
                summary["ready"],
                summary["error_count"],
                summary["warn_count"],
            )
        )
    return 0 if summary["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
