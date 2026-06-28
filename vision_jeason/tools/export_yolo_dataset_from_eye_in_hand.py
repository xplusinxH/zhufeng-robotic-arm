"""从 eye-in-hand 调试样本导出 YOLO 数据集骨架。

该工具只整理 ``color.png`` 为 YOLO 训练目录，并为每张图片创建空标签文件。
真实框由人工标注工具填写；这样可以避免继续信任 SAM2 这类自动预标注结果。
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CLASSES = ["beer_can", "earbud_case", "phone", "power_bank", "remote"]


def export_yolo_dataset(
    source_root,
    output_root,
    classes=None,
    split_name="train",
):
    """把采集样本整理为 YOLO images/labels 目录结构。"""

    source_root = Path(source_root)
    output_root = Path(output_root)
    classes = list(classes or DEFAULT_CLASSES)
    image_dir = output_root / "images" / split_name
    label_dir = output_root / "labels" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for color_path in sorted(source_root.glob("*/color.png")):
        frame_name = color_path.parent.name
        image_name = frame_name + ".png"
        label_name = frame_name + ".txt"
        shutil.copyfile(str(color_path), str(image_dir / image_name))
        (label_dir / label_name).touch()
        records.append(
            {
                "frame": frame_name,
                "source_image": str(color_path),
                "image": str(image_dir / image_name),
                "label": str(label_dir / label_name),
            }
        )

    dataset_yaml = {
        "path": str(output_root),
        "train": "images/{0}".format(split_name),
        "val": "images/{0}".format(split_name),
        "names": {index: name for index, name in enumerate(classes)},
    }
    _write_simple_yaml(output_root / "dataset.yaml", dataset_yaml)
    (output_root / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (output_root / "source_frames.json").write_text(
        json.dumps({"frames": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "image_count": len(records),
        "class_count": len(classes),
        "output_root": str(output_root),
    }


def _write_simple_yaml(path, data):
    """写出 YOLO 可读的极简 YAML，避免额外依赖。"""

    lines = [
        "path: {0}".format(data["path"]),
        "train: {0}".format(data["train"]),
        "val: {0}".format(data["val"]),
        "names:",
    ]
    for index, name in data["names"].items():
        lines.append("  {0}: {1}".format(index, name))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_classes(value):
    """解析逗号分隔类别名。"""

    if not value:
        return DEFAULT_CLASSES
    classes = [item.strip() for item in str(value).split(",") if item.strip()]
    if not classes:
        raise ValueError("类别列表不能为空")
    return classes


def main(argv=None):
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path, help="eye_in_hand_debug 数据目录")
    parser.add_argument("--output-root", required=True, type=Path, help="YOLO 数据集输出目录")
    parser.add_argument("--classes", help="逗号分隔类别名")
    parser.add_argument("--split-name", default="train", help="输出 split 名称，默认 train")
    args = parser.parse_args(argv)

    result = export_yolo_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        classes=parse_classes(args.classes),
        split_name=args.split_name,
    )
    print(
        "yolo_dataset={0} images={1} classes={2}".format(
            result["output_root"],
            result["image_count"],
            result["class_count"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
