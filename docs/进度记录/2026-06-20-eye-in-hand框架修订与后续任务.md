# 2026-06-20 Eye-in-hand 框架修订与后续任务记录

## 当前结论

已确认 D435 相机安装在机械臂末端执行器上，项目执行框架从“固定相机 + 相机坐标系桌面平面标定”修正为 eye-in-hand 方案。

当前正确链路为：

```text
控制侧发送 T_base_tool
手眼标定提供 T_tool_camera
Jetson 计算 T_base_camera = T_base_tool * T_tool_camera
深度点 P_camera 转换为 P_base
用 P_base.z 相对 Z_base = 0 判断桌面上方物体
```

桌面平面不再需要在相机坐标系中单独标定。机械臂安装基座底面所在平面就是桌面平面，在机械臂基坐标系中定义为 `Z_base = 0`。

## 已完成事项

1. 新增控制侧位姿帧解析模块 `communication/pose_protocol.py`。
2. 新增坐标变换组合模块 `coordinate/frame_transform.py`。
3. 重写 `perception/table_segment.py`，改为基坐标系高度过滤。
4. 重写 `perception/object_fusion.py`，基于 `center_base_m` 输出未知物体候选。
5. 修改 `perception/object_cluster.py`，支持输出基坐标系中心点。
6. 删除固定相机方案下无用的桌面平面标定和检查入口。
7. 同步 `vision_jeason`，后续可直接拷贝到 Jetson 目标目录。
8. 新增执行框架说明文档 `docs/eye_in_hand_execution_framework.md`。

## 当前外部依赖

控制侧暂时还不能实时发送 `T_base_tool`，因此以下工作暂时不能完成端到端联调：

1. Jetson 主循环实时读取机械臂末端位姿。
2. 使用真实 `T_base_tool` 计算实时 `T_base_camera`。
3. 在机械臂运动过程中验证 `P_base.z` 高度过滤稳定性。
4. 完整验证视觉目标坐标输出到控制侧后的闭环效果。

该依赖不阻塞视觉端继续开发。当前应把它作为控制侧接口等待项，而不是暂停整个项目。

## 可继续推进的视觉端任务

1. 增加 `tool_camera.yaml` 配置读取模块，先用模拟手眼外参占位。
2. 实现离线 eye-in-hand 场景回放测试，用静态深度图和模拟 `T_base_tool` 验证坐标链路。
3. 完善候选物输出数据结构，统一使用 `center_base_m`、`bbox_pixel`、`score`、`source`。
4. 改造调试工具，使其可以在无控制侧串口输入时使用手动输入或文件输入的 `T_base_tool`。
5. 设计后续串口输入协议的缓存层，等控制侧准备好后只替换数据源。
6. 继续补充工程级中文注释，保证后续迁移到 Jetson 时可读、可维护。

## 建议下一步

优先做“不依赖控制侧”的视觉端基础设施：

```text
新增 tool_camera.yaml 读取
新增 T_base_tool 文件/手动输入模拟器
新增离线 eye-in-hand 调试脚本
跑 PC 端测试
再同步 vision_jeason
```

这样控制侧暂时不能发送位姿时，视觉端仍然可以验证完整数学链路；等控制侧串口就绪后，只需要把模拟输入替换成真实串口输入。

## 2026-06-20 继续推进记录

已开始执行“不依赖控制侧”的视觉端任务。

新增内容：

1. `calibration/tool_camera_io.py`：读取和保存 `tool_camera.yaml` 手眼外参文件。
2. `communication/pose_source.py`：从手动文件读取模拟 `T_base_tool` 位姿帧。
3. `tools/offline_eye_in_hand_debug.py`：离线运行 eye-in-hand 几何链路，输出基坐标系候选物结果。
4. `calibration/tool_camera.example.yaml`：手眼外参占位示例文件。
5. `tools/base_tool_pose.example.txt`：模拟控制侧位姿帧示例文件。

当前离线验证链路：

```text
tool_camera.yaml
base_tool_pose.txt
depth.json
intrinsics.json
        ↓
offline_eye_in_hand_debug.py
        ↓
T_base_camera
base 坐标系候选物 JSON
```

后续控制侧串口可用后，`pose_source.py` 的文件输入可以替换为真实串口输入，后续感知链路无需重写。

## 2026-06-20 单帧真实相机调试入口

继续推进“不依赖控制侧”的现场调试能力，新增 `tools/capture_eye_in_hand_debug.py`。

该工具用于 Jetson 端真实 D435 单帧调试：

1. 启动 D435。
2. 采集一帧 aligned depth。
3. 读取 aligned depth 内参。
4. 从 `tool_camera.yaml` 读取手眼外参 `T_tool_camera`。
5. 从手动位姿文件读取模拟 `T_base_tool`。
6. 计算 `T_base_camera`。
7. 输出基坐标系下的未知物体候选 JSON。

控制侧暂时不能发送 `T_base_tool` 时，现场可以先用：

```bash
python3 tools/capture_eye_in_hand_debug.py \
  --tool-camera calibration/tool_camera.example.yaml \
  --base-tool-pose tools/base_tool_pose.example.txt \
  --output-root /mnt/zhufeng_data/zhufeng/data/eye_in_hand_debug \
  --min-z-base 0.01 \
  --max-z-base 0.30
```

注意：`tool_camera.example.yaml` 只是占位示例，不是真实手眼标定结果；正式联调前必须替换为实测 `tool_camera.yaml`。

## 2026-06-21 串口协议改写

已确认旧版只发送几何中心的协议不足以支持真实抓取，因为几何中心不能描述不规则物体的可抓取区域。

当前协议调整为：

1. `@OBJ`：发送物体候选摘要，包括候选中心、图像 ROI、点数和来源。
2. `@GRASP`：发送抓取建议，包括抓取点、抓取姿态、夹爪开口、质量评分和接近方式。
3. `@NOOBJ#`：没有候选物。
4. `@ERR,code,message#`：视觉端异常。
5. `@END,count#`：多帧输出结束。

重要约定：

- `bbox_pixel` 只是图像 ROI，不代表物体形状。
- `OBJ` 帧只告诉控制端“哪里有候选物”。
- `GRASP` 帧才是控制端优先执行的抓取目标。
- 旧版 `@TARGET` 暂时保留为早期调试兼容接口，不作为正式抓取协议。

## 2026-06-21 相机视野优先约束

根据当前机械结构图，D435 相机安装在夹爪上方。抓取过程中夹爪、被抓物体或末端结构可能遮挡相机视线，因此后续抓取逻辑必须优先保证相机视角不被遮挡。

新增约束：

1. 抓取候选必须先通过相机视野安全检查。
2. 会在预抓取阶段遮挡目标 ROI 的候选应直接拒绝。
3. 夹爪闭合前应尽量保持目标在相机视野中，便于最后一次定位确认。
4. `@GRASP` 协议新增 `visibility` 字段，表示相机视野安全评分。
5. 后续抓取评分必须把 `visibility_score` 作为优先级高于几何中心距离的因素。

更新后的抓取建议协议：

```text
@GRASP,id,x_mm,y_mm,z_mm,qx,qy,qz,qw,width_mm,quality,visibility,approach#
```

## 2026-06-21 第一版视野优先抓取建议生成

新增 `perception/grasp_planner.py`，把“相机视野优先”从协议字段推进为可测试的抓取候选筛选逻辑。

当前实现：

1. 使用 `camera_keepout_roi` 表示夹爪容易遮挡相机视野的图像区域。
2. 使用候选物 `bbox_pixel` 与 `camera_keepout_roi` 的重叠比例估算 `visibility`。
3. `visibility` 低于阈值时，不生成 `GRASP`。
4. `visibility` 达标时，生成第一版保守抓取建议，并把 `visibility` 写入抓取字典。

当前版本是启发式，不是最终抓取规划器。后续需要结合：

- 夹爪 CAD 投影
- 相机内参和 `T_tool_camera`
- 预抓取路径
- 点云法向和物体局部几何
- 夹爪开口与碰撞余量

把二维 ROI 视野评分升级为三维遮挡检测。

## 2026-06-21 单帧调试输出抓取建议

已将 `perception/grasp_planner.py` 接入离线和真实相机单帧调试脚本：

1. `tools/offline_eye_in_hand_debug.py`
2. `tools/capture_eye_in_hand_debug.py`

调试结果 JSON 现在不仅包含 `candidates`，还包含：

```text
grasp_count
rejected_grasp_count
grasps
```

新增 CLI 参数：

```text
--camera-keepout-roi u1,v1,u2,v2
--min-visibility 0.60
```

现场调试时可以先根据画面手动设置 `camera_keepout_roi`。当候选物 ROI 与该禁区重叠过多时，系统会保留候选物 `OBJ`，但不会生成 `GRASP`，从而避免控制端执行会遮挡相机的抓取动作。
