# Jetson Nano D435 视觉项目

本目录包含逐锋机械臂桌面整理项目的视觉端代码。电脑端用于开发和测试，
Jetson Nano 用于连接 RealSense D435、运行视觉程序和与下位机通信。

## 当前运行环境

- Jetson Nano B01
- L4T R32.7.4 / JetPack 4.6.4
- Python 3.6.9
- OpenCV 3.2.0
- librealsense 2.50.0，使用 RSUSB 后端
- RealSense D435，序列号 `243122071071`

## Jetson 存储布局

eMMC 保存系统、运行库和关键配置：

```text
/etc/zhufeng-vision/config
/etc/zhufeng-vision/calibration
```

外部 30 GB 存储已格式化为 ext4，并固定挂载到：

```text
/mnt/zhufeng_data
```

项目运行目录：

```text
/mnt/zhufeng_data/zhufeng/vision_project
```

## 相机验证

在 Jetson 中运行：

```bash
cd /mnt/zhufeng_data/zhufeng/vision_project
python3 tools/check_camera.py --frames 30
```

程序会采集完成对齐的彩色帧和深度帧，并打印中心点深度。

十分钟稳定性测试：

```bash
timeout 600 python3 tools/check_camera.py
```

## 开发约束

- 不升级 JetPack、CUDA、TensorRT 或 Linux 内核。
- Jetson 代码必须兼容 Python 3.6。
- 配置和标定结果保存在 eMMC。
- 项目、模型、日志和实验数据保存在外部 ext4 存储。
- 后续所有 Markdown 文档使用中文。
