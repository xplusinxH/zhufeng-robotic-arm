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

## 6. 串口输出协议修订

旧版 `@TARGET,class,x,y,z,score#` 只能表达目标中心点，不能完整支持抓取任务。当前正式协议改为两层：

1. `@OBJ`：物体候选摘要，只描述候选区域和图像 ROI，不表示物体是长方体。
2. `@GRASP`：抓取建议，描述控制端真正需要执行的抓取点、姿态、夹爪开口和质量评分。

候选物帧：

```text
@OBJ,id,class,score,x_mm,y_mm,z_mm,u1,v1,u2,v2,point_count,source#
```

示例：

```text
@OBJ,3,unknown,0.82,215.0,-40.0,52.0,120,80,210,170,438,base_height#
```

抓取建议帧：

```text
@GRASP,id,x_mm,y_mm,z_mm,qx,qy,qz,qw,width_mm,quality,visibility,approach#
```

示例：

```text
@GRASP,3,218.0,-42.0,68.0,0.0000,0.7071,0.0000,0.7071,35.0,0.76,0.91,top#
```

其它控制帧：

```text
@NOOBJ#
@ERR,code,message#
@END,count#
```

`bbox_pixel` 只用于控制端或调试端理解图像候选区域，不能作为物体几何形状使用。控制端正式执行时应优先消费 `GRASP` 帧，而不是直接抓取 `OBJ` 中的几何中心。

## 7. 相机视野优先抓取约束

根据当前机械结构，D435 位于夹爪上方，且相机视线会穿过或靠近夹爪工作区域。因此抓取逻辑必须优先保证相机视角不被夹爪、物体或末端结构遮挡。

该约束高于“抓取点距离最近”和“几何中心最优”：

1. 抓取候选必须先通过相机视野安全检查，再进入质量评分。
2. 会在预抓取阶段遮挡目标 ROI 的抓取候选应直接拒绝。
3. 夹爪闭合前应尽量保持目标在相机视野中，便于最后一次位置确认。
4. `GRASP.visibility` 表示本次抓取建议的相机视野安全评分，范围为 `0.00` 到 `1.00`。
5. 后续抓取评分应优先选择 `visibility` 高的候选；当 `visibility` 低于阈值时，即使几何抓取点看似可行，也不应发送给控制端执行。

后续抓取规划应把评分拆成：

```text
grasp_quality      抓取几何质量
visibility_score   相机视野安全程度
execution_margin   与夹爪/桌面/机械结构的安全余量
```

正式发送给控制端的 `quality` 可以是综合评分，但必须保留单独的 `visibility` 字段用于调试和安全判断。

当前第一版实现位于 `perception/grasp_planner.py`：

```text
candidate bbox_pixel
camera_keepout_roi
        ↓
estimate_visibility_score()
        ↓
visibility < 阈值：拒绝该抓取
visibility >= 阈值：生成初始 GRASP 建议
```

该版本使用图像 ROI 与相机视野禁区的重叠比例作为启发式评分。它的作用是先把“不要挡住相机”变成可测试的工程规则；后续应使用夹爪 CAD 投影、相机内外参和预抓取路径做更精确的三维遮挡检测。

`tools/offline_eye_in_hand_debug.py` 和 `tools/capture_eye_in_hand_debug.py` 已接入该模块。调试输出 JSON 会包含：

```text
candidate_count
candidates
grasp_count
rejected_grasp_count
grasps
```

现场可通过以下参数调整第一版视野约束：

```text
--camera-keepout-roi u1,v1,u2,v2
--min-visibility 0.60
```

其中 `camera_keepout_roi` 表示夹爪或末端结构容易遮挡相机视野的图像区域。初期可人工根据调试画面估计，后续再由夹爪三维模型自动投影生成。
