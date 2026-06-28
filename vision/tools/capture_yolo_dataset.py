"""YOLO 训练图片采集工具。

该工具只负责快速采集 RGB 图片和类别元数据，不做检测、不做标注。现场操作时可以
一键保存当前画面，也可以一键切换当前物品类别，后续再把图片交给人工标注流程生成
YOLO 框标签。
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from camera.realsense_camera import RealSenseCamera
from tools.export_yolo_dataset_from_eye_in_hand import DEFAULT_CLASSES


DEFAULT_OUTPUT_ROOT = Path("/mnt/zhufeng_data/data/yolo_raw_capture")
WINDOW_NAME = "YOLO Capture"


class CaptureClassState:
    """采集时的当前类别状态。"""

    def __init__(self, classes):
        if not classes:
            raise ValueError("类别列表不能为空")
        self.classes = list(classes)
        self.index = 0

    def current_class(self):
        """返回当前类别编号和名称。"""

        return self.index, self.classes[self.index]

    def next_class(self):
        """切换到下一个类别。"""

        self.index = (self.index + 1) % len(self.classes)
        return self.current_class()

    def previous_class(self):
        """切换到上一个类别。"""

        self.index = (self.index - 1) % len(self.classes)
        return self.current_class()

    def select_class(self, index):
        """按类别编号直接选择类别。"""

        if index < 0 or index >= len(self.classes):
            raise ValueError("类别编号超出范围")
        self.index = int(index)
        return self.current_class()


def parse_classes(value):
    """解析命令行类别列表。"""

    if not value:
        return list(DEFAULT_CLASSES)
    classes = [item.strip() for item in str(value).split(",") if item.strip()]
    if not classes:
        raise ValueError("类别列表不能为空")
    return classes


def build_sample_record(
    output_root,
    class_id,
    class_name,
    image_width,
    image_height,
    timestamp=None,
):
    """生成单张采集图片的路径和元数据。"""

    timestamp = timestamp or _timestamp()
    image_path = Path(output_root) / "images" / str(class_name) / (timestamp + ".png")
    return {
        "captured_at": timestamp,
        "class_id": int(class_id),
        "class_name": str(class_name),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "image_path": str(image_path),
    }


def save_capture_metadata(output_root, record, classes):
    """追加写入采集元数据，并同步类别文件。"""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "metadata.jsonl").open("a", encoding="utf-8") as metadata_file:
        metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_root / "classes.txt").write_text(
        "\n".join(str(item) for item in classes) + "\n",
        encoding="utf-8",
    )


def save_current_frame(output_root, color_bgr, class_state, cv2_module):
    """保存当前彩色画面和类别元数据。"""

    class_id, class_name = class_state.current_class()
    image_height, image_width = color_bgr.shape[:2]
    record = build_sample_record(
        output_root=output_root,
        class_id=class_id,
        class_name=class_name,
        image_width=image_width,
        image_height=image_height,
    )
    image_path = Path(record["image_path"])
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(image_path), color_bgr):
        raise RuntimeError("图片保存失败：{0}".format(image_path))
    save_capture_metadata(output_root, record, class_state.classes)
    return record


def run_capture(
    output_root=DEFAULT_OUTPUT_ROOT,
    classes=None,
    width=640,
    height=480,
    fps=30,
):
    """启动 RealSense 实时采集窗口。"""

    import cv2
    import numpy as np

    camera = RealSenseCamera(width=width, height=height, fps=fps)
    class_state = CaptureClassState(classes or DEFAULT_CLASSES)
    saved_count = 0
    camera.start()
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        print("YOLO 图片采集已启动：S 保存，N/P 切换类别，1-9 直接选类别，Q/Esc 退出。")
        print("输出目录：{0}".format(Path(output_root)))
        while True:
            color_frame, _depth_frame = camera.capture_aligned()
            color_bgr = np.asanyarray(color_frame.get_data()).copy()
            _draw_overlay(color_bgr, class_state, saved_count, cv2)
            cv2.imshow(WINDOW_NAME, color_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                record = save_current_frame(output_root, color_bgr, class_state, cv2)
                saved_count += 1
                print("saved class={0} path={1}".format(record["class_name"], record["image_path"]))
            elif key in (ord("n"), ord("N")):
                class_id, class_name = class_state.next_class()
                print("class={0}:{1}".format(class_id, class_name))
            elif key in (ord("p"), ord("P")):
                class_id, class_name = class_state.previous_class()
                print("class={0}:{1}".format(class_id, class_name))
            elif ord("1") <= key <= ord("9"):
                target_index = key - ord("1")
                if target_index < len(class_state.classes):
                    class_id, class_name = class_state.select_class(target_index)
                    print("class={0}:{1}".format(class_id, class_name))
    finally:
        camera.stop()
        cv2.destroyAllWindows()


def _draw_overlay(image_bgr, class_state, saved_count, cv2_module):
    """在预览画面上显示当前类别和操作提示。"""

    class_id, class_name = class_state.current_class()
    text = "class {0}: {1} | saved {2} | S save | N/P class | Q quit".format(
        class_id + 1,
        class_name,
        int(saved_count),
    )
    cv2_module.rectangle(image_bgr, (0, 0), (image_bgr.shape[1], 32), (0, 0, 0), -1)
    cv2_module.putText(
        image_bgr,
        text,
        (10, 22),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2_module.LINE_AA,
    )


def _timestamp():
    """生成文件名安全的毫秒时间戳。"""

    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]


def main(argv=None):
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--classes", help="逗号分隔类别名")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)

    run_capture(
        output_root=args.output_root,
        classes=parse_classes(args.classes),
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
