# Jetson Nano + RealSense D435 桌面整理视觉定位项目执行文档

> **项目定位**：本项目面向桌面整理任务，用户负责视觉端。Jetson Nano 采集 RealSense D435 的 RGB-D 数据，完成桌面物体检测、三维坐标解算、坐标系转换，并通过串口将目标类别、坐标和状态发送至另一块主控板。机械臂运动控制、轨迹规划和抓取执行不在本项目范围内。

---

## 0. 文档维护规则

### 0.1 主控原则

本 Markdown 文件作为项目后续执行的主控文档。后续所有方案变更、模块增删、参数调整、串口协议修改、标定方法修改、模型替换和实验结论，都应同步记录到本文档中。

### 0.2 后续修改要求

每次修改应至少记录：

```text
修改日期
修改内容
修改原因
影响模块
是否需要重新标定
是否影响串口协议
```

### 0.3 当前版本

| 项目 | 内容 |
|---|---|
| 文档版本 | v1.3 |
| 创建日期 | 2026-05-28 |
| 当前方案 | 标准版本 |
| 运行平台 | Jetson Nano B01 |
| 相机 | Intel RealSense D435 |
| 任务 | 桌面整理视觉定位 |
| 输出方式 | 串口发送坐标给下位机 |

---

# 1. 项目目标

## 1.1 总目标

构建一套可在 Jetson Nano 上运行的桌面物体视觉定位系统，实现：

```text
桌面场景 RGB-D 采集
桌面平面识别与去除
桌面物体分割与聚类
轻量目标检测 / 分类
目标三维坐标计算
相机坐标系到工作坐标系转换
串口发送目标类别、坐标和状态
```

## 1.2 项目边界

本项目负责：

```text
1. D435 相机接入
2. RGB-D 图像采集
3. 相机内参与深度参数读取
4. RGB 与 Depth 对齐
5. 桌面平面标定
6. 相机坐标系到工作坐标系标定
7. 桌面物体候选区域提取
8. 轻量目标识别或分类
9. 目标三维坐标计算
10. 串口协议设计与发送
11. 视觉端日志记录与调试
```

本项目不负责：

```text
1. 机械臂逆运动学
2. 机械臂运动规划
3. 夹爪控制
4. 电机控制
5. 抓取轨迹生成
6. 下位机控制策略
7. 急停与电机安全逻辑
```

---

# 2. 硬件与存储条件

## 2.1 当前硬件

| 模块 | 型号 / 条件 |
|---|---|
| 边缘计算平台 | Jetson Nano B01 |
| 系统存储 | Yahboom 套件自带 16GB eMMC |
| 当前 eMMC 剩余空间 | 约 8GB |
| 扩展存储 | 32GB 高速 SD 卡 |
| 深度相机 | Intel RealSense D435 |
| 输出通信 | 串口 UART / USB-TTL |
| 下位机 | 另一块主控板，负责机械臂控制 |

## 2.2 存储使用原则

eMMC 只放：

```text
系统
核心依赖
RealSense SDK
OpenCV
Python 基础库
串口通信库
```

SD 卡放：

```text
项目代码
模型文件
标定文件
日志
测试图像
少量实验数据
```

不建议在 Nano 上保存：

```text
大量 RGB-D 视频
大型训练数据集
SAM / SAM3 / 大模型权重
多个 Python 虚拟环境
大型源码编译缓存
```

推荐目录：

```text
/mnt/sdcard/vision_project
/mnt/sdcard/vision_project/models
/mnt/sdcard/vision_project/calibration
/mnt/sdcard/vision_project/logs
/mnt/sdcard/vision_project/test_data
```

---

# 3. 标准版本总体方案

## 3.1 技术路线

标准版本采用：

```text
RealSense D435
   ↓
RGB-D 对齐采集
   ↓
相机与工作坐标系标定
   ↓
桌面平面检测 / 标定
   ↓
桌面平面去除
   ↓
物体候选区域聚类
   ↓
轻量模型检测 / 分类
   ↓
目标中心点与三维坐标计算
   ↓
SE(3) / Sim(3) 坐标转换
   ↓
串口发送目标信息
```

## 3.2 为什么选择该方案

桌面整理任务中，目标具有泛化性。桌面上可能出现未知物体，单纯依赖 YOLO 检测容易漏检。因此标准版本采用：

```text
深度几何负责“找出桌面上的独立物体”
轻量模型负责“识别或粗分类”
坐标转换负责“把相机坐标变成下位机能用的坐标”
串口协议负责“稳定输出给控制板”
```

核心思想：

```text
不要求识别所有物体名称，但必须稳定输出可整理物体的三维位置。
```

---

# 4. 软件模块划分

推荐项目结构：

```text
vision_project/
├── main.py                    # 主程序入口
├── config.yaml                # 全局参数配置
├── requirements.txt           # Python 依赖
│
├── camera/
│   ├── realsense_camera.py    # D435 采集、对齐、内参读取
│   └── depth_utils.py         # 深度滤波、深度有效性判断
│
├── calibration/
│   ├── camera_intrinsic.json  # 相机内参记录
│   ├── extrinsic_se3.yaml     # 相机到工作坐标系 SE(3) 参数
│   ├── sim3.yaml              # 可选 Sim(3) 参数
│   ├── table_plane.yaml       # 桌面平面参数
│   ├── calibrate_intrinsic.py # 内参检查脚本
│   ├── calibrate_extrinsic.py # 外参标定脚本
│   └── calibrate_table.py     # 桌面平面标定脚本
│
├── perception/
│   ├── table_segment.py       # 桌面平面检测与去除
│   ├── object_cluster.py      # 点云/深度聚类
│   ├── detector_light.py      # YOLOv5n / SSD / MobileNet 等轻量识别
│   └── object_fusion.py       # 几何候选与识别结果融合
│
├── coordinate/
│   ├── pixel_to_3d.py         # 像素 + 深度 → 相机三维坐标
│   ├── transform.py           # SE(3) / Sim(3) 坐标变换
│   └── filter_coord.py        # 多帧坐标滤波
│
├── communication/
│   ├── serial_sender.py       # 串口发送
│   └── protocol.py            # 串口协议封装与解析辅助
│
├── tools/
│   ├── check_storage.sh       # 存储检查
│   ├── check_camera.py        # 相机检查
│   ├── check_serial.py        # 串口检查
│   └── visualize_debug.py     # 调试显示
│
├── models/
│   └── README.md              # 模型文件说明
│
├── logs/
│   └── README.md              # 日志说明
│
└── docs/
    └── experiment_record.md   # 实验记录
```

---

# 5. 标定流程

标定是本项目的关键模块，必须在正式识别和串口输出前完成。

## 5.1 标定总览

本项目至少需要完成四类标定：

| 标定类型 | 是否必须 | 目的 |
|---|---:|---|
| D435 内参读取与记录 | 必须 | 像素坐标转三维坐标 |
| RGB-Depth 对齐验证 | 必须 | 保证检测框和深度值对应 |
| 桌面平面标定 | 必须 | 提取桌面上方物体 |
| 相机坐标系到工作坐标系外参标定 | 必须 | 发送下位机可用坐标 |
| 深度尺度修正 / Sim(3) | 可选 | 修正整体比例误差 |

---

## 5.2 D435 内参读取

### 目的

获得：

```text
fx
fy
cx
cy
depth_scale
image_width
image_height
```

### 输出文件

```text
calibration/camera_intrinsic.json
```

### 示例内容

```json
{
  "width": 640,
  "height": 480,
  "fx": 615.0,
  "fy": 615.0,
  "cx": 320.0,
  "cy": 240.0,
  "depth_scale": 0.001,
  "unit": "meter"
}
```

### 验收标准

```text
1. 能正常读取 color intrinsics
2. 能正常读取 depth intrinsics
3. depth_scale 不为空
4. RGB 与 Depth 使用同一分辨率或已完成对齐
```

---

## 5.3 RGB 与 Depth 对齐验证

### 目的

D435 的 RGB 图像和深度图来自不同传感器，必须确认深度已经对齐到彩色图。

### 操作要求

程序中必须使用：

```text
align depth to color
```

### 验证方法

将鼠标放在彩色图目标中心位置，读取同一点的深度值，观察：

```text
目标中心深度是否合理
目标边缘深度是否出现明显错位
移动物体时深度区域是否跟随物体移动
```

### 验收标准

```text
1. 彩色图目标中心处能读取有效深度
2. 目标边缘深度与 RGB 轮廓基本一致
3. 深度图无明显左右偏移
```

---

## 5.4 桌面平面标定

### 目的

确定桌面在相机坐标系下的平面方程，用于过滤桌面本身，只保留桌面上方物体。

平面方程：

```text
ax + by + cz + d = 0
```

### 推荐方法

方法一：RANSAC 自动拟合桌面平面。

```text
1. 采集空桌面 RGB-D 数据
2. 生成点云
3. 在工作区域内用 RANSAC 拟合最大平面
4. 保存平面参数 a,b,c,d
```

方法二：手动选取桌面区域点云拟合平面。

### 输出文件

```text
calibration/table_plane.yaml
```

### 示例内容

```yaml
plane:
  a: 0.01
  b: -0.98
  c: 0.20
  d: 0.45
height_range:
  min_above_table_m: 0.01
  max_above_table_m: 0.30
```

### 使用规则

只保留桌面上方：

```text
1 cm ~ 30 cm
```

范围内的点。

### 验收标准

```text
1. 空桌面时不会产生大量假目标
2. 放置物体后能保留物体点云
3. 桌面边缘、背景墙、远处物体被明显抑制
```

---

## 5.5 相机坐标系到工作坐标系外参标定

### 目的

将 D435 相机坐标转换为下位机需要的工作坐标。

相机坐标：

```text
P_cam = [Xc, Yc, Zc]
```

工作坐标：

```text
P_work = [Xw, Yw, Zw]
```

刚体变换：

```text
P_work = R · P_cam + t
```

其中：

```text
R：3×3 旋转矩阵
t：3×1 平移向量
```

### 标定方式

推荐使用多点对应标定。

准备至少 4 个不共面的标定点，实际更推荐 6~10 个点。

每个点需要记录：

```text
1. 相机测得坐标 P_cam
2. 下位机 / 工作台定义坐标 P_work
```

示例：

| 点编号 | Xc | Yc | Zc | Xw | Yw | Zw |
|---|---:|---:|---:|---:|---:|---:|
| P1 | ... | ... | ... | ... | ... | ... |
| P2 | ... | ... | ... | ... | ... | ... |
| P3 | ... | ... | ... | ... | ... | ... |
| P4 | ... | ... | ... | ... | ... | ... |

### 求解方法

优先使用 SE(3)：

```text
P_work = R · P_cam + t
```

如果存在明显尺度误差，再使用 Sim(3)：

```text
P_work = s · R · P_cam + t
```

### 输出文件

```text
calibration/extrinsic_se3.yaml
```

示例：

```yaml
R:
  - [1.0, 0.0, 0.0]
  - [0.0, 1.0, 0.0]
  - [0.0, 0.0, 1.0]
t:
  - 0.0
  - 0.0
  - 0.0
unit: "mm"
rms_error_mm: 0.0
```

可选 Sim(3)：

```yaml
scale: 1.0
R:
  - [1.0, 0.0, 0.0]
  - [0.0, 1.0, 0.0]
  - [0.0, 0.0, 1.0]
t:
  - 0.0
  - 0.0
  - 0.0
unit: "mm"
rms_error_mm: 0.0
```

### 验收标准

```text
1. 标定点平均误差 ≤ 10 mm，基础可用
2. 标定点平均误差 ≤ 5 mm，较好
3. 坐标轴方向与下位机定义一致
4. 单位统一，建议串口输出使用 mm
```

---

## 5.6 深度尺度检查

### 目的

确认 D435 深度值是否存在系统性偏差。

### 方法

用尺子或已知距离物体测试：

```text
实际距离 200 mm
实际距离 300 mm
实际距离 400 mm
实际距离 500 mm
```

记录 D435 测得距离。

### 判断

如果误差近似为固定比例，例如全部偏大 3%，可考虑 Sim(3) 中加入尺度因子：

```text
s = 实际距离 / 测量距离
```

如果误差随机波动，应优先使用：

```text
深度中值滤波
异常值去除
多帧坐标滤波
```

不要盲目使用 Sim(3)。

---

# 6. 感知流程

## 6.1 D435 采集

基础设置：

```text
color: 640×480
depth: 640×480
fps: 30
align: depth to color
```

正式运行时建议关闭不必要的显示窗口。

---

## 6.2 深度预处理

必须处理：

```text
0 深度值
异常远点
异常近点
孤立噪声
边缘跳变
```

推荐策略：

```text
1. 限制有效深度范围，例如 0.15 m ~ 1.20 m
2. 对目标区域使用 7×7 或 11×11 深度中值
3. 去除 0 值
4. 去除超过中位数一定比例的异常点
5. 连续 5 帧坐标做中值滤波
```

---

## 6.3 桌面平面去除

输入：

```text
对齐后的 depth
相机内参
table_plane.yaml
```

输出：

```text
桌面上方有效点云
```

保留规则：

```text
点到桌面距离 > 10 mm
点到桌面距离 < 300 mm
点在预设工作区域内
```

---

## 6.4 物体聚类

输入：

```text
桌面上方点云
```

方法：

```text
DBSCAN
连通域分析
欧式聚类
```

输出：

```text
object candidates
```

每个候选物体包含：

```text
bbox_2d
center_pixel
center_3d_cam
size_3d
mask / region
valid_depth_ratio
```

过滤规则：

```text
1. 面积太小，过滤
2. 高度太低，过滤
3. 深度有效像素太少，过滤
4. 超出工作区域，过滤
```

---

## 6.5 轻量检测 / 分类

标准版本可选模型：

```text
SSD-Mobilenet-v2
YOLOv5n
YOLOv8n
MobileNet 分类器
```

推荐类别不要过细，先定义为：

```text
book
paper
bottle
cup
tool
electronic
unknown
```

若模型无法判断类别，保留：

```text
unknown
```

桌面整理项目中，unknown 仍然可以作为可整理物体输出。

---

## 6.6 几何结果与识别结果融合

融合规则：

```text
1. 深度聚类负责生成物体候选区域
2. 轻量模型负责提供类别和置信度
3. 如果模型检测框与候选区域 IoU 较高，则合并类别
4. 如果没有匹配类别，则该候选区域标记 unknown
```

这样可以避免模型漏检导致物体完全丢失。

---

# 7. 三维坐标计算

## 7.1 像素坐标转相机坐标

对于目标中心点：

```text
u, v
```

读取深度：

```text
Z
```

使用公式：

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = Z
```

得到：

```text
P_cam = [X, Y, Z]
```

单位建议在内部统一为 meter，串口输出统一转为 mm。

---

## 7.2 目标深度取值规则

禁止直接只取单点深度：

```text
Z = depth[u, v]
```

必须使用邻域统计：

```text
1. 以目标中心为中心取 7×7 或 11×11 区域
2. 去除 0 值
3. 去除异常值
4. 取中值作为 Z
```

对于较大物体，可在候选 mask 内取深度中位数。

---

## 7.3 坐标滤波

推荐连续 5 帧坐标滤波：

```text
最近 5 帧 X 取中值
最近 5 帧 Y 取中值
最近 5 帧 Z 取中值
```

若目标丢失，不应立即发送错误坐标，应发送：

```text
@NO_TARGET#
```

或保持上一帧坐标但标记状态为：

```text
STALE
```

---

# 8. 坐标系转换

## 8.1 默认使用 SE(3)

```text
P_work = R · P_cam + t
```

## 8.2 可选使用 Sim(3)

当存在整体尺度误差时：

```text
P_work = s · R · P_cam + t
```

默认：

```text
s = 1.0
```

即退化为 SE(3)。

## 8.3 坐标轴约定

必须与下位机约定：

```text
X 正方向
Y 正方向
Z 正方向
坐标原点
单位
是否需要发送姿态角
```

当前标准版本仅发送：

```text
类别 + 目标中心三维坐标
```

不发送抓取姿态。

---

# 9. 串口通信协议

## 9.1 基础设置

初始建议：

```text
baudrate: 115200
data bits: 8
stop bits: 1
parity: none
encoding: ascii
```

## 9.2 单目标协议

检测到目标：

```text
@TARGET,class,x,y,z,score#
```

示例：

```text
@TARGET,bottle,126.4,-35.2,280.7,0.86#
```

未检测到目标：

```text
@NO_TARGET#
```

## 9.3 多目标协议

多个目标时：

```text
@OBJ,id,class,x,y,z,w,h,score#
@OBJ,id,class,x,y,z,w,h,score#
@END,count#
```

示例：

```text
@OBJ,1,bottle,126.4,-35.2,280.7,52.0,132.0,0.86#
@OBJ,2,book,-80.0,35.0,302.5,180.0,22.0,0.79#
@END,2#
```

## 9.4 状态字段

后续可扩展：

```text
OK
NO_TARGET
LOW_CONF
BAD_DEPTH
OUT_OF_RANGE
STALE
CAMERA_ERROR
```

## 9.5 下位机联调要求

必须联合验证：

```text
1. 下位机能识别帧头 @
2. 下位机能识别帧尾 #
3. 坐标正负号解析正确
4. 小数解析正确
5. 多目标连续发送不会粘包误判
6. NO_TARGET 能被正确处理
```

---

# 10. 开发阶段计划

## 阶段 1：硬件与环境检查

目标：

```text
确认 Jetson Nano 存储、相机、串口可用
```

检查命令：

```bash
df -h
lsblk
lsusb
python3 --version
```

验收：

```text
1. eMMC 剩余空间 ≥ 3GB
2. SD 卡可挂载
3. D435 可被识别
4. 串口设备可见，例如 /dev/ttyUSB0 或 /dev/ttyTHS1
```

---

## 阶段 2：D435 采集与对齐

目标：

```text
完成 RGB-D 实时采集，并将 depth 对齐到 color
```

验收：

```text
1. 能显示 RGB 图
2. 能显示 Depth 图
3. 鼠标选点可读取深度
4. RGB 与 Depth 基本对齐
```

---

## 阶段 3：标定

目标：

```text
完成内参记录、桌面平面标定、相机到工作坐标系外参标定
```

输出：

```text
camera_intrinsic.json
table_plane.yaml
extrinsic_se3.yaml
```

验收：

```text
1. 标定文件齐全
2. 坐标轴方向正确
3. 坐标单位统一为 mm 输出
4. 标定误差满足项目要求
```

---

## 阶段 4：桌面物体候选提取

目标：

```text
通过桌面去除和点云聚类找到桌面上的独立物体
```

验收：

```text
1. 空桌面无明显假目标
2. 放置 1 个物体能稳定识别 1 个候选
3. 放置多个物体能输出多个候选
4. 候选中心点稳定
```

---

## 阶段 5：轻量检测 / 分类

目标：

```text
对候选物体进行粗分类
```

验收：

```text
1. 已知类别可输出类别名
2. 未知物体输出 unknown
3. 不因模型漏检而完全丢失几何候选
```

---

## 阶段 6：三维坐标与坐标转换

目标：

```text
输出工作坐标系下的目标中心坐标
```

验收：

```text
1. 静止目标坐标抖动较小
2. 不同位置目标坐标方向正确
3. 实测误差可接受
```

建议指标：

```text
基础可用：平均定位误差 ≤ 15 mm
较好效果：平均定位误差 ≤ 10 mm
优秀效果：平均定位误差 ≤ 5 mm
```

---

## 阶段 7：串口通信联调

目标：

```text
将目标信息稳定发送给下位机
```

验收：

```text
1. 下位机能稳定接收
2. 坐标数值解析正确
3. 无目标时能收到 NO_TARGET
4. 多目标时能正确识别数量
5. 长时间运行无明显断连
```

---

## 阶段 8：系统集成与演示

目标：

```text
完成桌面整理视觉端演示
```

演示内容：

```text
1. 桌面上放置多个物体
2. 系统识别桌面物体候选区域
3. 输出类别或 unknown
4. 计算三维坐标
5. 串口发送给下位机
6. 下位机根据坐标执行后续动作
```

---

# 11. 模型选择原则

## 11.1 当前不采用

不建议 Jetson Nano 本地部署：

```text
SAM3
GroundingDINO
大型 VLM
YOLO-World
大型分割网络
大规模开放词汇识别模型
```

原因：

```text
1. Jetson Nano 算力不足
2. 4GB 内存限制明显
3. 模型依赖重
4. 实时性差
5. 存储空间紧张
6. 工程风险高
```

## 11.2 推荐采用

标准版本可采用：

```text
OpenCV 几何分割
SSD-Mobilenet-v2
YOLOv5n
YOLOv8n
MobileNet 分类器
```

## 11.3 大模型的合理使用方式

SAM3 / 大模型可作为：

```text
1. PC 端离线自动标注工具
2. 后续高算力平台升级方向
3. 报告中的扩展方案
```

不作为当前 Nano 实时主流程。

---

# 12. 实验记录要求

每次实验记录：

```text
日期
场景
目标数量
目标类别
相机高度
相机角度
光照条件
平均误差
最大误差
是否丢目标
串口是否正常
问题描述
修改建议
```

实验记录写入：

```text
docs/experiment_record.md
```

---

# 13. 风险与应对

## 13.1 深度噪声大

应对：

```text
1. 目标区域深度中值
2. 多帧坐标滤波
3. 限制有效深度范围
4. 调整相机角度
5. 避免强反光物体
```

## 13.2 桌面平面误检

应对：

```text
1. 重新采集空桌面标定数据
2. 限制工作区域 ROI
3. 调整 RANSAC 阈值
4. 增加高度过滤
```

## 13.3 多物体粘连

应对：

```text
1. 调整聚类距离阈值
2. 使用 RGB 边缘辅助分割
3. 让下位机优先处理最清晰目标
4. 增加目标间距作为演示约束
```

## 13.4 类别识别不稳定

应对：

```text
1. 保留 unknown 类
2. 减少类别数量
3. 使用几何候选兜底
4. PC 端扩充训练数据
```

## 13.5 串口通信异常

应对：

```text
1. 增加帧头帧尾
2. 增加状态码
3. 降低发送频率
4. 增加 ACK 机制
5. 检查波特率一致性
```

---

# 14. 当前执行清单

## 14.1 立即执行

```text
1. 确认 SD 卡挂载路径
2. 创建 vision_project 项目目录
3. 检查 D435 是否能被系统识别
4. 检查 pyrealsense2 是否可用
5. 检查串口设备是否可见
6. 建立 config.yaml
7. 编写 D435 RGB-D 采集脚本
```

## 14.2 第二步执行

```text
1. 读取并保存相机内参
2. 验证 RGB-Depth 对齐
3. 采集空桌面数据
4. 标定桌面平面
5. 采集标定点
6. 求解相机到工作坐标系外参
```

## 14.3 第三步执行

```text
1. 实现桌面平面去除
2. 实现物体聚类
3. 实现目标中心三维坐标计算
4. 实现坐标滤波
5. 实现串口发送
```

## 14.4 第四步执行

```text
1. 接入轻量检测 / 分类模型
2. 几何候选与类别融合
3. 多目标串口协议测试
4. 完成桌面整理视觉端演示
```

---

# 15. 修订记录

| 日期 | 版本 | 修改内容 | 是否影响标定 | 是否影响串口协议 |
|---|---|---|---|---|
| 2026-05-28 | v1.0 | 创建标准版项目执行文档，加入标定、RGB-D 几何分割、轻量检测、坐标转换和串口协议 | 是 | 是 |

---

# 16. 当前最终方案摘要

当前项目采用标准版本：

```text
D435 RGB-D 采集
→ RGB-Depth 对齐
→ 相机内参记录
→ 桌面平面标定
→ 相机到工作坐标系外参标定
→ 桌面平面去除
→ 桌面物体聚类
→ 轻量检测 / 分类
→ 三维坐标计算
→ SE(3) / Sim(3) 坐标变换
→ 串口发送给下位机
```

当前方案不将 SAM3 作为 Jetson Nano 实时主流程。SAM3 仅作为 PC 离线标注、复杂场景增强或后续高算力平台升级方向。

---

# 17. 编程与开发方式

## 17.1 推荐结论

本项目推荐采用：

```text
主力电脑 VS Code 编写与管理代码
Jetson Nano 终端运行、测试、调试
必要时使用 VS Code Remote-SSH 连接 Jetson Nano
```

不推荐长期直接在 Jetson Nano 本机桌面运行完整 VS Code 作为主力 IDE。

## 17.2 推荐开发模式

### 模式 A：电脑端 VS Code + SSH 远程连接 Jetson Nano

推荐程度：最高。

流程：

```text
电脑上使用 VS Code
通过 SSH 连接 Jetson Nano
代码实际存放在 Jetson Nano 的 SD 卡项目目录
在 VS Code 中远程编辑
在 Jetson Nano 终端中运行程序
```

优点：

```text
1. 编辑体验好
2. 不占用 Nano 桌面资源
3. 可以直接运行 Nano 上的 Python 环境
4. 方便管理项目文件
5. 适合长期开发
```

建议项目路径：

```text
/mnt/sdcard/vision_project
```

### 模式 B：电脑端写代码，拷贝到 Nano 运行

推荐程度：高。

适合网络不稳定或暂时不想配置 Remote-SSH 的情况。

流程：

```text
电脑端 VS Code 写代码
使用 scp / rsync / U 盘同步到 Nano
Nano 终端运行测试
```

优点：

```text
1. 简单
2. 不依赖 VS Code 远程插件
3. Nano 负载低
```

缺点：

```text
1. 文件同步需要手动管理
2. 调试效率略低
```

### 模式 C：Nano 本机终端 + nano/vim 编写

推荐程度：中。

适合临时修改配置或小脚本。

可用工具：

```text
nano
vim
gedit
tmux
bash
python3
```

适合修改：

```text
config.yaml
串口号
阈值参数
简单测试脚本
```

不适合大规模代码开发。

### 模式 D：Nano 本机安装 VS Code

推荐程度：低。

原因：

```text
1. Jetson Nano 内存只有 4GB
2. VS Code 图形界面占资源
3. 同时运行 D435、OpenCV、Python 程序时容易卡顿
4. eMMC 空间紧张
5. 对本项目收益不高
```

仅在必须本机图形化编辑时考虑。

## 17.3 本项目最终采用规则

当前标准版本采用：

```text
电脑端 VS Code 作为主要代码编辑器
Jetson Nano 作为运行与测试平台
Nano 上主要使用终端执行程序
必要时使用 VS Code Remote-SSH
```

Nano 上正式运行程序时，建议关闭不必要的桌面窗口和调试显示，优先保证 D435 采集、坐标计算和串口通信稳定。

## 17.4 常用终端命令

进入项目：

```bash
cd /mnt/sdcard/vision_project
```

运行主程序：

```bash
python3 main.py
```

查看串口设备：

```bash
ls /dev/ttyUSB*
ls /dev/ttyTHS*
```

查看相机设备：

```bash
lsusb
```

查看存储：

```bash
df -h
lsblk
```

## 17.5 推荐安装工具

Nano 上建议保留轻量工具：

```bash
sudo apt update
sudo apt install -y git vim nano tmux htop
```

如使用 SSH：

```bash
sudo apt install -y openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh
```

## 17.6 修订记录补充

| 日期 | 版本 | 修改内容 | 是否影响标定 | 是否影响串口协议 |
|---|---|---|---|---|
| 2026-05-28 | v1.1 | 增加编程与开发方式：推荐电脑端 VS Code + Jetson Nano 终端/Remote-SSH，不推荐 Nano 本机长期运行完整 VS Code | 否 | 否 |

---

# 18. Jetson Nano 依赖版本原则

## 18.1 核心结论

本项目所有 CUDA、cuDNN、TensorRT、PyTorch、OpenCV、RealSense SDK 等依赖，必须以 Jetson Nano 当前系统的 JetPack / L4T 版本为基准，不允许直接按普通 Ubuntu 或 x86 电脑的最新版安装。

## 18.2 原因

Jetson Nano 的 GPU 驱动、CUDA、cuDNN、TensorRT 与 JetPack / L4T 深度绑定。错误安装不匹配版本可能导致：

```text
1. CUDA 不可用
2. PyTorch 无法调用 GPU
3. TensorRT 导入失败
4. OpenCV 与系统库冲突
5. RealSense 编译或运行异常
6. 系统依赖被破坏
7. eMMC 空间被大量占用
```

## 18.3 本项目优先原则

依赖安装优先级：

```text
第一优先级：系统已有 JetPack 自带组件
第二优先级：NVIDIA 官方 Jetson 对应版本包
第三优先级：Jetson 社区已验证 wheel / deb 包
第四优先级：源码编译
```

不建议：

```text
1. 直接 pip install 最新 torch
2. 直接安装 CUDA 11 / CUDA 12
3. 直接使用普通 Ubuntu x86 教程
4. 随意升级系统大版本
5. 在 Nano 上源码编译大型 OpenCV / PyTorch
```

## 18.4 版本检查命令

查看 L4T 版本：

```bash
cat /etc/nv_tegra_release
```

查看 CUDA：

```bash
nvcc --version
```

查看 TensorRT：

```bash
dpkg -l | grep nvinfer
```

查看 cuDNN：

```bash
dpkg -l | grep cudnn
```

查看 OpenCV：

```bash
python3 -c "import cv2; print(cv2.__version__)"
```

查看 Python：

```bash
python3 --version
```

查看 RealSense：

```bash
rs-enumerate-devices
```

查看 pyrealsense2：

```bash
python3 -c "import pyrealsense2 as rs; print(rs)"
```

## 18.5 Jetson Nano 常见版本基线

若当前系统为 JetPack 4.6.x / L4T R32.7.x，常见组件基线通常为：

```text
CUDA 10.2
cuDNN 8.2.x
TensorRT 8.2.x
OpenCV 4.1.x
Ubuntu 18.04 系列环境
```

实际安装必须以本机命令输出为准。

## 18.6 本项目推荐环境策略

标准版本优先采用：

```text
Python 3
pyrealsense2
OpenCV
NumPy
PySerial
轻量检测模型
```

只有在确实需要 GPU 推理时，再安装与 JetPack 匹配的 PyTorch / TensorRT。

## 18.7 对当前项目的影响

当前项目主流程为：

```text
D435 采集
桌面平面分割
物体聚类
轻量检测 / 分类
坐标计算
串口发送
```

其中前五项可以优先用 CPU + OpenCV + NumPy 完成。CUDA 并不是第一阶段必须项。若后续接入 YOLOv5n / YOLOv8n / SSD-Mobilenet，则再根据 JetPack 版本选择合适的 PyTorch 或 TensorRT 环境。

## 18.8 修订记录补充

| 日期 | 版本 | 修改内容 | 是否影响标定 | 是否影响串口协议 |
|---|---|---|---|---|
| 2026-05-28 | v1.2 | 增加 Jetson Nano 依赖版本原则：CUDA、cuDNN、TensorRT、PyTorch 等必须按 JetPack/L4T 匹配，不按普通 Ubuntu 最新版安装 | 否 | 否 |

---

# 19. 当前板卡实际系统版本记录

## 19.1 已确认版本

根据当前 Jetson Nano 终端输出：

```text
cat /etc/nv_tegra_release
# R32 (release), REVISION: 7.4, GCID: 33514132, BOARD: t210ref, EABI: aarch64, DATE: Fri Jun  9 04:25:08 UTC 2023
```

确认：

```text
L4T / Jetson Linux：R32.7.4
对应 JetPack：4.6.4
内核分支：4.9.337-tegra
架构：aarch64
板卡类型：t210ref，Jetson Nano 系列
```

## 19.2 已确认 NVIDIA L4T 组件

系统中已存在：

```text
nvidia-l4t-core
nvidia-l4t-cuda
nvidia-l4t-kernel
nvidia-l4t-camera
nvidia-l4t-multimedia
nvidia-l4t-gstreamer
nvidia-l4t-apt-source
```

说明该系统不是普通 Ubuntu，而是 NVIDIA Jetson L4T 系统。

## 19.3 nvcc 当前状态

当前执行：

```bash
nvcc --version
```

结果：

```text
bash: nvcc: command not found
```

这表示当前系统中暂时无法直接调用 CUDA 编译器。可能原因：

```text
1. cuda-toolkit 未完整安装
2. nvcc 所在目录没有加入 PATH
3. Yahboom 镜像只预装了运行时组件，没有安装完整 CUDA 编译工具链
```

## 19.4 对项目的影响

当前标准版本第一阶段不依赖 nvcc。

以下模块仍可优先推进：

```text
D435 采集
RGB-Depth 对齐
OpenCV 图像处理
桌面平面分割
物体聚类
坐标计算
串口发送
```

只有后续需要：

```text
CUDA 自定义编译
GPU 版 PyTorch
TensorRT 模型转换
CUDA 加速推理
```

时，才需要进一步修复或安装 nvcc / CUDA Toolkit。

## 19.5 后续依赖安装基准

后续所有 Jetson 相关依赖均以：

```text
L4T R32.7.4 / JetPack 4.6.4
```

为基准选择版本。

禁止直接安装：

```text
CUDA 11
CUDA 12
最新版 PyTorch
普通 x86 Ubuntu 教程中的深度学习环境
```

## 19.6 下一步检查命令

继续确认 CUDA 目录：

```bash
ls /usr/local/
ls -l /usr/local/cuda*
```

确认 CUDA 相关包：

```bash
dpkg -l | grep -E "cuda|cudnn|nvinfer|tensorrt"
```

确认 OpenCV：

```bash
python3 -c "import cv2; print(cv2.__version__)"
```

确认 RealSense：

```bash
rs-enumerate-devices
python3 -c "import pyrealsense2 as rs; print(rs)"
```

## 19.7 修订记录补充

| 日期 | 版本 | 修改内容 | 是否影响标定 | 是否影响串口协议 |
|---|---|---|---|---|
| 2026-05-28 | v1.3 | 记录当前板卡实际系统版本：L4T R32.7.4 / JetPack 4.6.4；确认 nvcc 暂不可用，但不影响第一阶段视觉流程 | 否 | 否 |
| 2026-05-28 | v1.4 | 建立电脑端开发环境与 Jetson 运行端环境边界：电脑端仅用于 VS Code/SSH 编程、轻量测试与静态检查；Jetson 端负责 RealSense、CUDA/TensorRT、串口等真实硬件依赖检查与运行。创建项目骨架、PC/Jetson 分离依赖文件、Jetson 环境检查脚本和基础设置脚本。 | 否 | 否 |
