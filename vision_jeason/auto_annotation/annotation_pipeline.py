"""eye-in-hand 数据集自动预标注流水线。

该模块只负责把分割后端输出的 mask 统一转换为评估工具需要的
``annotations.json``。具体使用 SAM2、SAM3、传统算法或测试后端，由外部 backend 决定。
"""

import json
from pathlib import Path


class MaskPrediction:
    """单个分割结果。

    ``mask_pixels`` 使用 ``[(u, v), ...]`` 存储前景像素，便于从 SAM2 mask、OpenCV 轮廓
    或测试后端统一转换为 bbox。
    """

    def __init__(self, mask_pixels, score=1.0, class_name="object"):
        self.mask_pixels = list(mask_pixels)
        self.score = float(score)
        self.class_name = str(class_name)


def annotate_eye_in_hand_dataset(
    dataset_root,
    output_annotations,
    backend,
    preview_writer=None,
    min_area_pixel=20,
    max_bbox_area_ratio=0.80,
    conservative_filter=True,
):
    """遍历 eye_in_hand_debug 目录并生成自动预标注 JSON。"""

    dataset_root = Path(dataset_root)
    frames = {}
    object_count = 0
    for color_path in sorted(dataset_root.glob("*/color.png")):
        frame_dir = color_path.parent
        image_size = _load_image_size(color_path)
        objects = []
        for prediction in backend.segment_image(color_path):
            if len(prediction.mask_pixels) < int(min_area_pixel):
                continue
            bbox = bbox_from_mask_pixels(prediction.mask_pixels)
            if bbox is None:
                continue
            if _is_rejected_large_bbox(bbox, image_size, max_bbox_area_ratio):
                continue
            objects.append(
                {
                    "bbox_pixel": bbox,
                    "class_name": prediction.class_name,
                    "score": prediction.score,
                }
            )
        if conservative_filter:
            objects = filter_obvious_objects(objects, image_size)
        frames[frame_dir.name] = {"objects": objects}
        object_count += len(objects)
        if preview_writer is not None:
            preview_writer.write_preview(color_path, objects, frame_dir / "annotation_preview.png")

    annotations = {
        "schema": "zhufeng_eye_in_hand_detection_annotations_v1",
        "source": "auto_annotation",
        "bbox_format": "[u1, v1, u2, v2]",
        "frames": frames,
    }
    output_annotations = Path(output_annotations)
    output_annotations.parent.mkdir(parents=True, exist_ok=True)
    output_annotations.write_text(
        json.dumps(annotations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "frame_count": len(frames),
        "object_count": object_count,
        "output_annotations": str(output_annotations),
    }


def bbox_from_mask_pixels(mask_pixels):
    """从 mask 像素集合计算 ``[u1, v1, u2, v2]``。"""

    pixels = list(mask_pixels)
    if not pixels:
        return None
    us = [int(pixel[0]) for pixel in pixels]
    vs = [int(pixel[1]) for pixel in pixels]
    return [min(us), min(vs), max(us), max(vs)]


def filter_obvious_objects(objects, image_size):
    """保守筛选自动标注候选，只保留明显像独立物体的 bbox。

    SAM2 全自动分割会同时给出整图背景、物体局部纹理和互相嵌套的 mask。用于生成
    评估标注时，宁可少留，也不能把大量碎片写进 ground truth。
    """

    if image_size is None:
        return list(objects)
    image_width, image_height = image_size
    image_area = float(image_width * image_height)
    kept = []
    for obj in sorted(objects, key=lambda item: _bbox_area(item["bbox_pixel"]), reverse=True):
        bbox = obj["bbox_pixel"]
        width = float(bbox[2] - bbox[0] + 1)
        height = float(bbox[3] - bbox[1] + 1)
        area_ratio = _bbox_area(bbox) / image_area
        width_ratio = width / float(image_width)
        height_ratio = height / float(image_height)
        if area_ratio < 0.01:
            continue
        if area_ratio > 0.35:
            continue
        if width_ratio < 0.08 or height_ratio < 0.08:
            continue
        if width_ratio > 0.75 or height_ratio > 0.75:
            continue
        if _touches_image_border(bbox, image_width, image_height, margin_ratio=0.03):
            continue
        if any(_bbox_iou(bbox, existing["bbox_pixel"]) > 0.55 for existing in kept):
            continue
        if any(_contained_ratio(bbox, existing["bbox_pixel"]) > 0.80 for existing in kept):
            continue
        kept.append(obj)
    kept.sort(key=lambda item: item["bbox_pixel"])
    return kept


def _load_image_size(image_path):
    """读取图像尺寸；测试伪文件或缺少 Pillow 时返回 None。"""

    try:
        from PIL import Image

        with Image.open(str(image_path)) as image:
            return image.size
    except Exception:
        return None


def _bbox_area(bbox):
    u1, v1, u2, v2 = [float(value) for value in bbox]
    return max(0.0, u2 - u1 + 1.0) * max(0.0, v2 - v1 + 1.0)


def _bbox_intersection_area(left, right):
    left_u1, left_v1, left_u2, left_v2 = [float(value) for value in left]
    right_u1, right_v1, right_u2, right_v2 = [float(value) for value in right]
    u1 = max(left_u1, right_u1)
    v1 = max(left_v1, right_v1)
    u2 = min(left_u2, right_u2)
    v2 = min(left_v2, right_v2)
    return max(0.0, u2 - u1 + 1.0) * max(0.0, v2 - v1 + 1.0)


def _bbox_iou(left, right):
    intersection = _bbox_intersection_area(left, right)
    union = _bbox_area(left) + _bbox_area(right) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _contained_ratio(inner, outer):
    inner_area = _bbox_area(inner)
    if inner_area <= 0.0:
        return 0.0
    return _bbox_intersection_area(inner, outer) / inner_area


def _touches_image_border(bbox, image_width, image_height, margin_ratio=0.015):
    margin = max(2, int(round(min(image_width, image_height) * float(margin_ratio))))
    u1, v1, u2, v2 = [int(value) for value in bbox]
    return (
        u1 <= margin
        or v1 <= margin
        or u2 >= int(image_width) - 1 - margin
        or v2 >= int(image_height) - 1 - margin
    )


def _is_rejected_large_bbox(bbox, image_size, max_bbox_area_ratio):
    """过滤 SAM 自动分割常见的整图背景 mask。"""

    if image_size is None:
        return False
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        return False
    u1, v1, u2, v2 = [int(value) for value in bbox]
    bbox_area = max(0, u2 - u1 + 1) * max(0, v2 - v1 + 1)
    image_area = int(image_width) * int(image_height)
    return float(bbox_area) / float(image_area) > float(max_bbox_area_ratio)
