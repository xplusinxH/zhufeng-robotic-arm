"""PC 端 eye-in-hand 调试数据自动预标注工具。

该工具读取每个时间戳目录中的 ``color.png``，通过可插拔分割后端生成物体 mask，
再转换为检测评估使用的 ``annotations.json``。第一版优先服务当前精度评估任务，
后续可在同一数据结构上扩展 YOLO bbox/segmentation 导出。
"""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from auto_annotation.annotation_pipeline import annotate_eye_in_hand_dataset
from auto_annotation.backends.dummy_backend import DummyBackend
from auto_annotation.backends.sam2_backend import Sam2Backend
from auto_annotation.preview_writer import OpenCvPreviewWriter


def build_backend(args):
    """根据命令行参数创建自动预标注分割后端。"""

    if args.backend == "dummy":
        return DummyBackend()
    if args.backend == "sam2":
        if not args.sam2_checkpoint or not args.sam2_model_cfg:
            raise ValueError("使用 sam2 后端时必须提供 --sam2-checkpoint 和 --sam2-model-cfg")
        return Sam2Backend(
            checkpoint_path=args.sam2_checkpoint,
            model_cfg=args.sam2_model_cfg,
            device=args.device,
            min_mask_area_pixel=args.min_area_pixel,
        )
    raise ValueError("未知自动标注后端：{0}".format(args.backend))


def main(argv=None):
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path, help="eye_in_hand_debug 数据目录")
    parser.add_argument("--output", type=Path, help="annotations.json 输出路径")
    parser.add_argument("--backend", choices=("dummy", "sam2"), default="sam2", help="自动分割后端")
    parser.add_argument("--sam2-checkpoint", help="SAM2 checkpoint 文件路径")
    parser.add_argument("--sam2-model-cfg", help="SAM2 模型配置名或配置路径")
    parser.add_argument("--device", default="cpu", help="SAM2 推理设备，当前 PC 默认使用 cpu")
    parser.add_argument("--min-area-pixel", type=int, default=80, help="保留 mask 的最小像素面积")
    parser.add_argument("--no-preview", action="store_true", help="不输出 annotation_preview.png")
    args = parser.parse_args(argv)

    output = args.output or (args.dataset_root / "annotations.json")
    backend = build_backend(args)
    preview_writer = None if args.no_preview else OpenCvPreviewWriter()
    result = annotate_eye_in_hand_dataset(
        dataset_root=args.dataset_root,
        output_annotations=output,
        backend=backend,
        preview_writer=preview_writer,
        min_area_pixel=args.min_area_pixel,
    )
    print(
        "auto_annotations={0} frames={1} objects={2}".format(
            result["output_annotations"],
            result["frame_count"],
            result["object_count"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
