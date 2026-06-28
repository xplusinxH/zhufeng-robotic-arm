# 桌面抓取视觉主程序

本程序入口为 `tools/run_desktop_vision.py`，默认使用当前手工标注训练得到的 `models/yolov8n_manual_best.pt`。如果后续在 Jetson 上导出了 `models/yolov8n_manual_best.engine`，程序会自动优先使用 TensorRT engine。

单帧检测命令：

```bash
cd /mnt/zhufeng_data/vision_jeason && python3 tools/run_desktop_vision.py
```

实时预览命令：

```bash
cd /mnt/zhufeng_data/vision_jeason && python3 tools/run_desktop_vision.py --show
```

实时预览按键：

- `D`：对当前帧执行一次 YOLO + 深度几何检测。
- `S`：保存当前相机画面。
- `Q` 或 `Esc`：退出程序。

控制端协议输出命令：

```bash
cd /mnt/zhufeng_data/vision_jeason && python3 tools/run_desktop_vision.py --protocol
```

默认输出目录为 `/mnt/zhufeng_data/data/yolo_depth_scene`。每次检测会生成 `desktop_vision_result.json`，包含识别类别、置信度、像素框、相机坐标、base 坐标、尺寸估计和抓取建议。
