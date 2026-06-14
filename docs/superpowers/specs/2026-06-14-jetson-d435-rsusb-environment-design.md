# Jetson Nano D435 RSUSB 环境设计

## 目标

在不修改 Jetson 内核的前提下，使 Jetson Nano 能够通过一台 Intel
RealSense D435 采集完成对齐的 RGB-D 图像。

## 已确认的硬件与平台

- Jetson Nano B01，运行 L4T R32.7.4 / JetPack 4.6.4。
- Python 3.6.9。
- Intel RealSense D435 已通过 USB 3.0 识别，设备编号为 `8086:0b07`。
- 根文件系统剩余空间约 8 GB。
- 外部 30 GB 存储挂载于 `/media/jetson/1896-8302`。
- 已发现串口设备 `/dev/ttyTHS1` 和 `/dev/ttyTHS2`。

## 技术方案

使用 librealsense 的 RSUSB 后端。该方案不需要应用内核补丁，是现有
JetPack 4.6.4 环境中风险最低的实现路径。

通过 Ubuntu 软件包安装轻量级系统依赖和 OpenCV。在外部存储中克隆并编译
librealsense，编译配置如下：

- 启用 RSUSB 后端。
- 为系统 Python 3.6 启用 Python 绑定。
- 除非诊断需要，否则不编译示例和图形工具。
- 使用适合 Jetson Nano 内存条件的保守并行编译参数。

本阶段不安装或升级 CUDA、TensorRT、PyTorch、Linux 内核或 JetPack。

## 存储布局

- librealsense 源码与构建目录：
  `/media/jetson/1896-8302/src/librealsense`
- 视觉项目部署目录：
  `/media/jetson/1896-8302/vision_project`
- 系统库和 Python 绑定：
  在成功编译后安装至标准系统路径

## 执行阶段

1. 更新软件包索引，安装基础编译工具和系统版 OpenCV。
2. 验证 Python 能够导入 OpenCV。
3. 克隆兼容版本的 librealsense，并配置 RSUSB 构建。
4. 编译并安装 librealsense 与 Python 绑定。
5. 验证设备枚举与 `pyrealsense2` 导入。
6. 实现并运行 RGB-D 对齐和深度读取冒烟测试。

每个阶段通过验收后才能进入下一阶段。

## 异常处理

- 软件包安装出现依赖冲突时立即停止。
- 根文件系统剩余空间低于 3 GB 时立即停止。
- 所有编译产物保存在外部存储中。
- 使用较低的并行编译数量，降低内存不足风险。
- 不以内核补丁作为失败后的备用方案。

## 验收标准

- `python3 -c "import cv2; print(cv2.__version__)"` 执行成功。
- `rs-enumerate-devices` 能够检测到 D435。
- `python3 -c "import pyrealsense2"` 执行成功。
- 测试程序能够以 640x480 分辨率采集彩色图和深度图。
- 深度图成功对齐到彩色图。
- 选定的彩色图像素能够返回合理的深度值。
- 采集测试连续运行至少 10 分钟且不崩溃。

## 延后工作

- CUDA、TensorRT 和 PyTorch 环境配置。
- 目标检测与分类。
- 桌面平面标定与分割。
- 机械臂运动和串口集成。
