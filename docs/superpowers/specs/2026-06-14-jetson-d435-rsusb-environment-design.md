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

### eMMC：系统与关键配置

eMMC 当前容量约 14 GB，已使用约 5 GB。项目运行期间必须始终保留至少
4 GB 可用空间。

允许写入 eMMC 的内容：

- Ubuntu、JetPack 和系统软件包。
- OpenCV、librealsense 运行库及 Python 绑定。
- udev 规则和后续需要的 systemd 服务文件。
- 项目运行配置：`/etc/zhufeng-vision/config`。
- 相机内参、桌面平面和坐标变换等标定结果：
  `/etc/zhufeng-vision/calibration`。

eMMC 中的项目配置和标定文件总量应控制在 100 MB 内。日志、模型、源码、
构建产物和测试数据禁止写入 eMMC。

### 外部 30 GB 存储：项目工作盘

外部存储挂载于 `/media/jetson/1896-8302`，目录分配如下：

- 项目源码：`/media/jetson/1896-8302/zhufeng/vision_project`，预算 1 GB。
- 第三方源码与构建产物：`/media/jetson/1896-8302/zhufeng/build`，预算 8 GB。
- 模型文件：`/media/jetson/1896-8302/zhufeng/models`，预算 6 GB。
- 日志：`/media/jetson/1896-8302/zhufeng/logs`，预算 3 GB，后续启用轮转。
- 测试图像与实验数据：`/media/jetson/1896-8302/zhufeng/data`，预算 8 GB。
- 临时文件：`/media/jetson/1896-8302/zhufeng/tmp`，预算 2 GB，可随时清理。

至少保留 2 GB 外部存储空闲空间。运行程序通过配置路径读取 eMMC 中的配置和
标定文件，并将模型、日志和数据路径指向外部存储。

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
- 根文件系统剩余空间低于 4 GB 时立即停止。
- 外部存储未挂载时禁止启动正式视觉程序，避免数据回落到 eMMC。
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
