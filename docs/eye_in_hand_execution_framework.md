# Eye-in-hand 视觉执行框架修订

修改日期：2026-06-20

## 1. 关键纠偏

D435 相机安装在机械臂末端执行器上，属于 eye-in-hand 结构，因此相机相对桌面的位置和角度会随机械臂运动实时变化。

项目不再使用“固定相机 + 相机坐标系桌面平面标定”的执行框架。桌面平面由机械结构先验定义：机械臂安装基座底面所在平面就是桌面平面，在机械臂基坐标系中记为 `Z_base = 0`。

## 2. 当前运行链路

1. 控制侧通过串口向 Jetson 发送当前末端位姿 `T_base_tool`。
2. 手眼标定提供固定外参 `T_tool_camera`。
3. Jetson 计算当前相机位姿：

```text
T_base_camera = T_base_tool * T_tool_camera
```

4. D435 深度点先由像素坐标转换为相机坐标系点 `P_camera`。
5. Jetson 使用当前 `T_base_camera` 将点转换到机械臂基坐标系：

```text
P_base = T_base_camera * P_camera
```

6. 物体候选区域使用基坐标系高度过滤：

```text
min_z_base < P_base.z < max_z_base
```

7. 后续聚类、候选物输出和串口发送均以 `base` 坐标系结果为主。

## 3. 已删除的旧模块

以下模块基于固定相机和相机坐标系桌面平面假设，已经从当前工程删除：

- `vision/calibration/calibrate_table.py`
- `vision/tools/check_table_scene.py`
- `vision_jeason/calibration/calibrate_table.py`
- `vision_jeason/tools/check_table_scene.py`

## 4. 当前新增/保留模块职责

- `communication/pose_protocol.py`：解析控制侧发送的 `T_base_tool` 位姿帧。
- `coordinate/frame_transform.py`：构造、组合和应用齐次坐标变换。
- `perception/table_segment.py`：在 `base` 坐标系中按高度提取桌面上方点。
- `perception/object_cluster.py`：把候选点聚类成物体候选区域，并输出 `center_base_m`。
- `perception/object_fusion.py`：生成未知物体候选，作为后续识别和抓取目标的几何入口。

## 5. 下一步工程任务

1. 实现 Jetson 串口接收缓存，把控制侧发送的 `T_base_tool` 接入主循环。
2. 增加 `tool_camera.yaml` 读取逻辑，等待后续手眼标定结果填入。
3. 在主循环中实时计算 `T_base_camera`，替换所有固定外参调用。
4. 用真实 D435 深度帧验证 `Z_base` 高度过滤阈值，先建议从 `0.01m` 到 `0.30m` 开始调试。
5. 再接入候选物串口输出协议，发送 `center_base_m` 而不是相机坐标系点。
