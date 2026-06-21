# Calibration Files

本目录只保存当前 eye-in-hand 方案仍然需要的标定文件。

- `camera_intrinsic.json`：D435 内参和深度比例。
- `tool_camera.yaml`：手眼标定结果，即固定的 `T_tool_camera`。
- `sim3.yaml`：可选的深度尺度修正参数。

已废弃：

- `table_plane.yaml`：固定相机方案下的相机坐标系桌面平面参数，当前方案不再使用。
- `extrinsic_se3.yaml`：固定相机到工作坐标系外参，当前由实时 `T_base_tool * T_tool_camera` 替代。
