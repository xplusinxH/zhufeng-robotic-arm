"""Jetson YOLO 深度识别一键部署测试。

该脚本把部署现场最常用的步骤串起来：自检、必要时导出 TensorRT engine、真实测速、
单帧协议输出。它只应在 Jetson 真机上运行；PC 本地只做步骤编排的单元测试。
"""

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MODEL = PROJECT_ROOT / "models" / "yolov8n_first_best.pt"
DEFAULT_ENGINE = PROJECT_ROOT / "models" / "yolov8n_first_best.engine"


def build_deploy_test_steps(
    model_path=DEFAULT_MODEL,
    engine_path=DEFAULT_ENGINE,
    engine_exists=False,
    frames=100,
    imgsz=640,
):
    """生成部署测试步骤。

    返回值用于展示和测试，不直接执行。真实运行由 ``run_deploy_test`` 顺序调用这些
    命令，任一步失败都会停止，避免后续结果误导现场判断。
    """

    model_path = Path(model_path)
    engine_path = Path(engine_path)
    steps = [
        {
            "name": "check",
            "command": _command_text(
                "tools/check_yolo_depth_deploy.py",
                "--model",
                model_path,
                "--engine",
                engine_path,
            ),
        }
    ]
    if not engine_exists:
        steps.append(
            {
                "name": "export",
                "command": _command_text(
                    "tools/export_yolo_tensorrt.py",
                    "--model",
                    model_path,
                    "--imgsz",
                    int(imgsz),
                ),
            }
        )
    steps.extend(
        [
            {
                "name": "benchmark",
                "command": _command_text(
                    "tools/benchmark_yolo_depth_jetson.py",
                    "--model",
                    engine_path,
                    "--frames",
                    int(frames),
                    "--imgsz",
                    int(imgsz),
                ),
            },
            {
                "name": "scene",
                "command": _command_text(
                    "tools/run_yolo_depth_scene.py",
                    "--model",
                    engine_path,
                    "--imgsz",
                    int(imgsz),
                    "--protocol",
                ),
            },
        ]
    )
    return steps


def run_deploy_test(model_path=DEFAULT_MODEL, engine_path=DEFAULT_ENGINE, frames=100, imgsz=640):
    """在 Jetson 上顺序执行部署测试步骤。"""

    steps = build_deploy_test_steps(
        model_path=model_path,
        engine_path=engine_path,
        engine_exists=Path(engine_path).exists(),
        frames=frames,
        imgsz=imgsz,
    )
    for index, step in enumerate(steps, 1):
        print("[{0}/{1}] {2}: {3}".format(index, len(steps), step["name"], step["command"]))
        subprocess.run(step["command"].split(), cwd=str(PROJECT_ROOT), check=True)
    return steps


def _command_text(script_path, *args):
    """生成稳定的单行命令文本，便于日志复制和测试。"""

    parts = ["python3", str(script_path)]
    for arg in args:
        parts.append(str(arg))
    return " ".join(parts)


def main(argv=None):
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO .pt 权重路径")
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE, help="YOLO TensorRT engine 路径")
    parser.add_argument("--frames", type=int, default=100, help="benchmark 正式统计帧数")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 输入尺寸")
    parser.add_argument("--dry-run", action="store_true", help="只打印步骤，不执行")
    args = parser.parse_args(argv)

    steps = build_deploy_test_steps(
        model_path=args.model,
        engine_path=args.engine,
        engine_exists=args.engine.exists(),
        frames=args.frames,
        imgsz=args.imgsz,
    )
    if args.dry_run:
        for index, step in enumerate(steps, 1):
            print("[{0}/{1}] {2}: {3}".format(index, len(steps), step["name"], step["command"]))
        return 0
    run_deploy_test(
        model_path=args.model,
        engine_path=args.engine,
        frames=args.frames,
        imgsz=args.imgsz,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
