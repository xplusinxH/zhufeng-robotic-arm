"""自动预标注结果预览图输出。"""


class OpenCvPreviewWriter:
    """使用 OpenCV 在原图上绘制自动标注框。"""

    def __init__(self, cv2_module=None):
        self.cv2 = cv2_module or _import_cv2()

    def write_preview(self, image_path, objects, output_path):
        """保存带 bbox 的预览图，便于人工快速复核自动标注结果。"""

        image = self.cv2.imread(str(image_path))
        if image is None:
            return False
        for obj in objects:
            u1, v1, u2, v2 = [int(value) for value in obj["bbox_pixel"]]
            self.cv2.rectangle(image, (u1, v1), (u2, v2), (0, 255, 255), 2)
            self.cv2.putText(
                image,
                "{0} {1:.2f}".format(obj.get("class_name", "object"), float(obj.get("score", 0.0))),
                (u1, max(14, v1 - 6)),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                self.cv2.LINE_AA,
            )
        return bool(self.cv2.imwrite(str(output_path), image))


def _import_cv2():
    import cv2

    return cv2
