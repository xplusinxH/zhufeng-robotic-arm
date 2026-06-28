"""Eye-in-hand 检测结果精度评测工具。

该工具读取每个调试样本中的 ``eye_in_hand_candidates.json``，再与人工标注的
真实物体框做 IoU 匹配，输出 precision、recall、F1 和逐帧 TP/FP/FN。它只评测
候选物检测框，不评测后续 YOLO 分类或机械臂抓取成功率。
"""

import argparse
import json
from pathlib import Path


def evaluate_dataset(dataset_root, annotations_path, iou_threshold=0.50):
    """评测一个 eye_in_hand_debug 数据目录。"""

    dataset_root = Path(dataset_root)
    annotations = load_annotations(annotations_path)
    frame_results = []
    totals = {"tp": 0, "fp": 0, "fn": 0}

    for frame_name in sorted(annotations["frames"].keys()):
        expected_objects = annotations["frames"][frame_name].get("objects", [])
        detected_objects = load_detected_objects(dataset_root / frame_name)
        frame_result = evaluate_frame(
            frame_name,
            detected_objects,
            expected_objects,
            iou_threshold=iou_threshold,
        )
        frame_results.append(frame_result)
        totals["tp"] += frame_result["tp"]
        totals["fp"] += frame_result["fp"]
        totals["fn"] += frame_result["fn"]

    precision = _safe_divide(totals["tp"], totals["tp"] + totals["fp"])
    recall = _safe_divide(totals["tp"], totals["tp"] + totals["fn"])
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    return {
        "dataset_root": str(dataset_root),
        "annotations_path": str(annotations_path),
        "iou_threshold": float(iou_threshold),
        "frame_count": len(frame_results),
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "frames": frame_results,
    }


def build_annotation_template(dataset_root):
    """根据调试样本目录生成空标注模板，避免人工手写帧名出错。"""

    dataset_root = Path(dataset_root)
    frames = {}
    for result_path in sorted(dataset_root.glob("*/eye_in_hand_candidates.json")):
        frames[result_path.parent.name] = {"objects": []}
    return {
        "schema": "zhufeng_eye_in_hand_detection_annotations_v1",
        "bbox_format": "[u1, v1, u2, v2]",
        "frames": frames,
    }


def evaluate_frame(frame_name, detected_objects, expected_objects, iou_threshold=0.50):
    """评测单帧候选框与人工标注框的匹配情况。"""

    matches = []
    unmatched_detection_indexes = set(range(len(detected_objects)))
    unmatched_expected_indexes = set(range(len(expected_objects)))
    scored_pairs = []
    for detection_index, detection in enumerate(detected_objects):
        for expected_index, expected in enumerate(expected_objects):
            iou = bbox_iou(detection["bbox_pixel"], expected["bbox_pixel"])
            scored_pairs.append((iou, detection_index, expected_index))
    scored_pairs.sort(reverse=True)

    for iou, detection_index, expected_index in scored_pairs:
        if iou < float(iou_threshold):
            continue
        if detection_index not in unmatched_detection_indexes:
            continue
        if expected_index not in unmatched_expected_indexes:
            continue
        unmatched_detection_indexes.remove(detection_index)
        unmatched_expected_indexes.remove(expected_index)
        matches.append(
            {
                "detection_index": detection_index,
                "expected_index": expected_index,
                "iou": iou,
            }
        )

    false_positives = [detected_objects[index] for index in sorted(unmatched_detection_indexes)]
    false_negatives = [expected_objects[index] for index in sorted(unmatched_expected_indexes)]
    return {
        "frame": frame_name,
        "tp": len(matches),
        "fp": len(false_positives),
        "fn": len(false_negatives),
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def bbox_iou(left, right):
    """计算两个 ``[u1,v1,u2,v2]`` 框的交并比。"""

    left_u1, left_v1, left_u2, left_v2 = _normalize_bbox(left)
    right_u1, right_v1, right_u2, right_v2 = _normalize_bbox(right)
    inter_u1 = max(left_u1, right_u1)
    inter_v1 = max(left_v1, right_v1)
    inter_u2 = min(left_u2, right_u2)
    inter_v2 = min(left_v2, right_v2)
    inter_area = _bbox_area((inter_u1, inter_v1, inter_u2, inter_v2))
    union_area = _bbox_area((left_u1, left_v1, left_u2, left_v2)) + _bbox_area(
        (right_u1, right_v1, right_u2, right_v2)
    ) - inter_area
    return _safe_divide(inter_area, union_area)


def load_detected_objects(frame_dir):
    """读取单个调试样本目录中的候选物框。"""

    result_path = Path(frame_dir) / "eye_in_hand_candidates.json"
    if not result_path.exists():
        return []
    data = json.loads(result_path.read_text(encoding="utf-8"))
    return [
        {
            "id": candidate.get("id"),
            "bbox_pixel": candidate["bbox_pixel"],
            "source": candidate.get("source", "unknown"),
            "score": candidate.get("score", 0.0),
        }
        for candidate in data.get("candidates", [])
        if "bbox_pixel" in candidate
    ]


def load_annotations(annotations_path):
    """读取人工标注文件，并检查基本结构。"""

    data = json.loads(Path(annotations_path).read_text(encoding="utf-8-sig"))
    if "frames" not in data:
        raise ValueError("标注文件必须包含 frames 字段")
    return data


def save_evaluation_report(report, output_path):
    """保存评测报告 JSON。"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_annotation_template(template, output_path):
    """保存人工标注模板 JSON。"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(template, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _normalize_bbox(bbox):
    u1, v1, u2, v2 = [float(value) for value in bbox]
    return min(u1, u2), min(v1, v2), max(u1, u2), max(v1, v2)


def _bbox_area(bbox):
    u1, v1, u2, v2 = bbox
    return max(0.0, float(u2) - float(u1)) * max(0.0, float(v2) - float(v1))


def _safe_divide(numerator, denominator):
    denominator = float(denominator)
    if denominator == 0.0:
        return 0.0
    return float(numerator) / denominator


def main(argv=None):
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="eye_in_hand_debug 数据根目录")
    parser.add_argument("--annotations", help="人工标注 JSON")
    parser.add_argument("--iou-threshold", type=float, default=0.50, help="判定匹配的 IoU 阈值")
    parser.add_argument("--output-json", help="评测报告输出路径")
    parser.add_argument("--init-annotations", help="先生成空标注模板 JSON")
    args = parser.parse_args(argv)

    if args.init_annotations:
        template = build_annotation_template(args.dataset_root)
        save_annotation_template(template, args.init_annotations)
        print(
            "annotation_template={0} frames={1}".format(
                args.init_annotations,
                len(template["frames"]),
            )
        )
        return 0
    if not args.annotations:
        parser.error("--annotations is required unless --init-annotations is used")
    report = evaluate_dataset(
        dataset_root=args.dataset_root,
        annotations_path=args.annotations,
        iou_threshold=args.iou_threshold,
    )
    if args.output_json:
        save_evaluation_report(report, args.output_json)
    print(
        "frames={0} tp={1} fp={2} fn={3} precision={4:.3f} recall={5:.3f} f1={6:.3f}".format(
            report["frame_count"],
            report["tp"],
            report["fp"],
            report["fn"],
            report["precision"],
            report["recall"],
            report["f1"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
