# Jetson YOLO TensorRT 部署测试

本阶段目标是先实测 Jetson 视觉端是否能接受：D435 采集、YOLO TensorRT 推理、检测框内深度 ROI 点云几何、输出 3D 候选物坐标。

## 准备权重

把 PC 训练得到的权重放到 Jetson 项目内默认路径：

```bash
/mnt/zhufeng_data/vision_jeason/models/yolov8n_first_best.pt
```

## 安装依赖

如果 Jetson 已经有可用的 RealSense SDK、OpenCV、TensorRT 和 PyTorch，只安装项目侧 Python 依赖：

```bash
cd /mnt/zhufeng_data/vision_jeason && python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-jetson-yolo.txt
```

## 部署自检

自检只在 Jetson 真机上作为部署前检查使用，本地 PC 不验证串口、RealSense 或 TensorRT 环境：

```bash
cd /mnt/zhufeng_data/vision_jeason && python3 tools/check_yolo_depth_deploy.py
```

如果自检出现 `error`，先修复依赖或模型文件，不要继续跑 TensorRT 导出和 benchmark。`pyrealsense2` 是 D435 真机采集必需项，缺失时必须先安装 RealSense SDK 对应 Python 绑定。

## 一键部署测试

优先使用这一条命令跑完整链路：自检、缺少 engine 时导出 TensorRT、benchmark、单帧协议输出。

```bash
cd /mnt/zhufeng_data/vision_jeason && python3 tools/run_yolo_depth_deploy_test.py
```

如果需要先看会执行哪些步骤，不实际打开相机或导出 engine：

```bash
cd /mnt/zhufeng_data/vision_jeason && python3 tools/run_yolo_depth_deploy_test.py --dry-run
```

## 导出 TensorRT

TensorRT engine 必须在 Jetson 本机生成：

```bash
cd /mnt/zhufeng_data/vision_jeason && python3 tools/export_yolo_tensorrt.py
```

导出成功后默认生成：

```bash
/mnt/zhufeng_data/vision_jeason/models/yolov8n_first_best.engine
```

## 跑真实测速

默认读取 `best.engine`，连续采集 10 帧预热和 100 帧正式统计：

```bash
cd /mnt/zhufeng_data/vision_jeason && python3 tools/benchmark_yolo_depth_jetson.py
```

结果保存到：

```bash
/mnt/zhufeng_data/data/yolo_jetson_benchmark
```

重点看 JSON 里的 `summary.total_ms.p95`、`summary.yolo_ms.p95` 和 `summary.geometry_ms.p95`。如果 `yolo_ms` 仍然太高，下一步优先测试 `--imgsz 416` 或 `--imgsz 320`，而不是继续改深度几何。

## 单帧识别输出

导出 engine 后，先跑一次完整单帧链路，保存候选物和 GRASP JSON：

```bash
cd /mnt/zhufeng_data/vision_jeason && python3 tools/run_yolo_depth_scene.py
```

如果要直接查看后续串口会发送的协议帧：

```bash
cd /mnt/zhufeng_data/vision_jeason && python3 tools/run_yolo_depth_scene.py --protocol
```

## 启动串口服务

控制端需要先发送当前末端位姿：

```text
@POSE,x,y,z,qx,qy,qz,qw#
```

其中 `x/y/z` 单位是米，四元数顺序是 `qx,qy,qz,qw`。Jetson 返回：

```text
@POSE_OK#
```

然后控制端发送：

```text
@DETECT#
```

Jetson 会返回一组 `@OBJ`、`@GRASP` 和 `@END` 帧。启动服务：

```bash
cd /mnt/zhufeng_data/vision_jeason && python3 tools/serve_yolo_depth_serial.py --serial-port /dev/ttyUSB0 --baudrate 115200
```

串口服务不在 PC 本地验证；只在 Jetson 真机上结合控制端串口测试。

该脚本默认优先使用：

```bash
/mnt/zhufeng_data/vision_jeason/models/yolov8n_first_best.engine
```

如果 engine 还没生成，会自动回退到：

```bash
/mnt/zhufeng_data/vision_jeason/models/yolov8n_first_best.pt
```
