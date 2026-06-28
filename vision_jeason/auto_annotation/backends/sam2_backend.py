"""SAM2 自动分割后端。

本后端只在 PC 离线自动预标注时使用，不进入 Jetson 实时检测链路。
SAM2 依赖和模型文件较重，因此全部采用延迟导入，保证普通单元测试和 Jetson 脚本不受影响。
"""

from auto_annotation.annotation_pipeline import MaskPrediction


class Sam2Backend:
    """基于 SAM2 automatic mask generator 的图片分割后端。"""

    def __init__(
        self,
        checkpoint_path,
        model_cfg,
        device="cpu",
        min_mask_area_pixel=20,
    ):
        self.checkpoint_path = str(checkpoint_path)
        self.model_cfg = str(model_cfg)
        self.device = str(device)
        self.min_mask_area_pixel = int(min_mask_area_pixel)
        self._generator = None

    def segment_image(self, image_path):
        """对单张图片执行自动分割，返回统一的 MaskPrediction 列表。"""

        generator = self._get_generator()
        image = self._load_rgb_image(image_path)
        masks = generator.generate(image)
        predictions = []
        for item in masks:
            segmentation = item.get("segmentation")
            if segmentation is None:
                continue
            ys, xs = self._np.nonzero(segmentation)
            if len(xs) < self.min_mask_area_pixel:
                continue
            pixels = list(zip(xs.tolist(), ys.tolist()))
            predictions.append(
                MaskPrediction(
                    mask_pixels=pixels,
                    score=float(item.get("predicted_iou", item.get("stability_score", 1.0))),
                    class_name="object",
                )
            )
        return predictions

    def _get_generator(self):
        if self._generator is not None:
            return self._generator
        try:
            import numpy as np
            import torch
            from PIL import Image
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
        except Exception as exc:
            raise RuntimeError(
                "SAM2 依赖未安装完整，请先安装 torch、Pillow 和 facebookresearch/sam2。"
            ) from exc

        self._np = np
        self._Image = Image
        model = build_sam2(self.model_cfg, self.checkpoint_path, device=self.device)
        if self.device == "cpu":
            model = model.to(torch.device("cpu"))
        self._generator = SAM2AutomaticMaskGenerator(model)
        return self._generator

    def _load_rgb_image(self, image_path):
        image = self._Image.open(str(image_path)).convert("RGB")
        return self._np.array(image)
