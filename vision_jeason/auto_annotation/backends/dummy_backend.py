"""测试与环境自检用的轻量分割后端。"""

from auto_annotation.annotation_pipeline import MaskPrediction


class DummyBackend:
    """返回一个固定小 mask，用于验证自动标注文件链路是否跑通。"""

    def segment_image(self, image_path):
        _ = image_path
        mask_pixels = []
        for v in range(10, 22):
            for u in range(10, 22):
                mask_pixels.append((u, v))
        return [
            MaskPrediction(
                mask_pixels=mask_pixels,
                score=1.0,
                class_name="object",
            )
        ]
