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
