# YOLO 数据采集流程

当前阶段目标是采集真实训练图片，不运行 YOLO，不做自动标注。

## 启动采集

```bash
cd /mnt/zhufeng_data/vision_jeason && python3 tools/capture_yolo_dataset.py
```

默认输出目录：

```text
/mnt/zhufeng_data/data/yolo_raw_capture
```

## 操作按键

```text
S：保存当前画面
N：切换到下一个类别
P：切换到上一个类别
1-9：直接选择类别
Q 或 Esc：退出
```

默认类别：

```text
beer_can,earbud_case,phone,power_bank,remote
```

如果要临时指定类别：

```bash
cd /mnt/zhufeng_data/vision_jeason && python3 tools/capture_yolo_dataset.py --classes beer_can,earbud_case,phone,power_bank,remote
```

## 采集建议

第一轮先采 150-300 张，每类 30-50 张。每类都要包含中心、边缘、多角度、多物体同框、不同距离，以及夹爪左下角白色前端正常出现的画面。

采集结果会保存为：

```text
images/<类别>/<时间戳>.png
metadata.jsonl
classes.txt
```
