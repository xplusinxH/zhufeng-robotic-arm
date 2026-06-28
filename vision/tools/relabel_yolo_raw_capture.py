"""重命名 YOLO 原始采集数据类别目录。

采集阶段可能使用了临时类别名。本工具按确认后的真实物体类别重命名目录，并根据
实际图片文件重建 ``classes.txt`` 和 ``metadata.jsonl``，避免沿用 Jetson 绝对路径
或旧类别名。
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


FINAL_CLASSES = ["beer_can", "earbud_case", "phone", "power_bank", "remote"]
DEFAULT_CLASS_RENAME_MAP = {
    "remote": "beer_can",
    "power_bank": "earbud_case",
    "circuit_board": "remote",
    "box": "phone",
    "other_object": "power_bank",
}


def relabel_raw_capture_dataset(
    root,
    rename_map=None,
    final_classes=None,
):
    """按映射重命名原始采集目录，并重建元数据。"""

    root = Path(root)
    image_root = root / "images"
    rename_map = dict(rename_map or DEFAULT_CLASS_RENAME_MAP)
    final_classes = list(final_classes or FINAL_CLASSES)
    if not image_root.exists():
        raise FileNotFoundError("找不到 images 目录：{0}".format(image_root))

    temp_root = image_root / "__relabel_tmp__"
    if temp_root.exists():
        shutil.rmtree(str(temp_root))
    temp_root.mkdir(parents=True)
    try:
        for old_name, new_name in rename_map.items():
            old_dir = image_root / old_name
            if not old_dir.exists():
                continue
            temp_dir = temp_root / old_name
            shutil.move(str(old_dir), str(temp_dir))
            target_dir = image_root / new_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for image_path in sorted(temp_dir.glob("*.png")):
                shutil.move(str(image_path), str(target_dir / image_path.name))
        records = build_rebuilt_metadata_records(root, final_classes)
        _write_metadata(root, records, final_classes)
        return {
            "image_count": len(records),
            "class_count": len(final_classes),
            "classes": final_classes,
            "root": str(root),
        }
    finally:
        if temp_root.exists():
            shutil.rmtree(str(temp_root))


def build_rebuilt_metadata_records(root, classes):
    """根据实际图片文件重建元数据记录。"""

    root = Path(root)
    records = []
    for class_id, class_name in enumerate(classes):
        class_dir = root / "images" / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.glob("*.png")):
            records.append(
                {
                    "captured_at": image_path.stem,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "image_path": str(image_path),
                }
            )
    return records


def _write_metadata(root, records, classes):
    """写出重建后的类别文件和 JSONL 元数据。"""

    root = Path(root)
    (root / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    with (root / "metadata.jsonl").open("w", encoding="utf-8") as metadata_file:
        for record in records:
            metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv=None):
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="yolo_raw_capture 根目录")
    args = parser.parse_args(argv)

    result = relabel_raw_capture_dataset(args.root)
    print(
        "relabel_done root={0} images={1} classes={2}".format(
            result["root"],
            result["image_count"],
            ",".join(result["classes"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
