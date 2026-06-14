# Jetson Nano D435 RSUSB 环境实施计划

> **执行要求：** 使用 `superpowers:subagent-driven-development`（推荐）或
> `superpowers:executing-plans` 按任务逐项实施。所有步骤使用复选框跟踪。

**目标：** 在不修改 Jetson 内核的前提下，完成 D435 的 RGB-D 对齐采集与深度读取验证。

**架构：** Jetson 使用系统 Python 3.6 和系统版 OpenCV；librealsense 在外部存储中以
RSUSB 后端编译，并安装 Python 绑定。仓库中的相机模块负责封装采集，工具脚本负责硬件验收。

**技术栈：** JetPack 4.6.4、Python 3.6、OpenCV、librealsense RSUSB、pytest。

---

## 文件结构

- 修改 `vision/camera/realsense_camera.py`：封装 D435 启动、对齐采集和释放资源。
- 修改 `vision/tools/check_camera.py`：提供可直接在 Jetson 运行的相机冒烟测试。
- 新建 `vision/tests/test_realsense_camera.py`：使用假模块测试相机封装，不依赖真实硬件。
- 修改 `vision/README.md`：以中文记录 Jetson 安装、构建和验收命令。
- 修改 `vision/jetson_nano_d435_desktop_sorting_project (5).md`：记录本阶段执行结果。

### 任务 0：初始化存储布局

- [ ] **步骤 1：确认两个存储设备的剩余空间**

```bash
df -h / /media/jetson/1896-8302
```

预期：eMMC 可用空间大于 4 GB，外部存储可用空间大于 20 GB。

- [ ] **步骤 2：在 eMMC 创建关键配置目录**

```bash
sudo mkdir -p /etc/zhufeng-vision/config
sudo mkdir -p /etc/zhufeng-vision/calibration
sudo chown -R jetson:jetson /etc/zhufeng-vision
```

预期：`jetson` 用户能够读写项目配置和标定目录。

- [ ] **步骤 3：在外部存储创建项目工作目录**

```bash
mkdir -p /media/jetson/1896-8302/zhufeng/vision_project
mkdir -p /media/jetson/1896-8302/zhufeng/models
mkdir -p /media/jetson/1896-8302/zhufeng/logs
mkdir -p /media/jetson/1896-8302/zhufeng/data
mkdir -p /media/jetson/1896-8302/zhufeng/tmp
```

预期：所有目录均创建成功，且位于外部存储。

- [ ] **步骤 4：创建第一版运行配置**

```bash
cp /media/jetson/1896-8302/zhufeng/vision_project/config.yaml \
  /etc/zhufeng-vision/config/config.yaml
```

预期：部署项目后，运行配置位于 eMMC 的 `/etc/zhufeng-vision/config`。
若项目尚未部署，则在部署完成后执行本步骤。

### 任务 1：安装基础依赖并验证 OpenCV

- [ ] **步骤 1：更新软件包索引**

在 Jetson Xshell 终端运行：

```bash
sudo apt update
```

预期：命令完成，结尾没有依赖冲突或无法下载的软件源错误。

- [ ] **步骤 2：安装最小基础依赖**

```bash
sudo apt install -y git cmake build-essential pkg-config libusb-1.0-0-dev \
  libssl-dev libgtk-3-dev python3-dev python3-pip python3-opencv
```

预期：安装成功，不删除 NVIDIA L4T 软件包。

- [ ] **步骤 3：验证 OpenCV 与磁盘空间**

```bash
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
df -h /
```

预期：打印 OpenCV 版本，根文件系统剩余空间大于 4 GB。

### 任务 2：在 eMMC 临时目录准备 librealsense RSUSB 源码

- [ ] **步骤 1：创建外部存储源码目录**

```bash
mkdir -p /home/jetson/build
cd /home/jetson/build
```

预期：当前目录为 `/home/jetson/build`。

- [ ] **步骤 2：克隆固定版本源码**

```bash
git clone --branch v2.50.0 --depth 1 https://github.com/IntelRealSense/librealsense.git
cd librealsense
```

预期：`git describe --tags` 输出 `v2.50.0`。

- [ ] **步骤 3：安装设备权限规则**

```bash
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

预期：命令无错误；完成后重新插拔 D435。

### 任务 3：配置、编译并安装 librealsense

- [ ] **步骤 1：创建构建目录**

```bash
cd /home/jetson/build/librealsense
mkdir -p build
cd build
```

- [ ] **步骤 2：配置 RSUSB 构建**

```bash
cmake .. \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF \
  -DBUILD_UNIT_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release
```

预期：配置成功，并显示 RSUSB 与 Python 绑定已启用。

- [ ] **步骤 3：低并行度编译**

```bash
make -j2
```

预期：编译完成。若发生内存不足，则改用 `make -j1`，不增加交换分区。

- [ ] **步骤 4：安装并刷新动态库缓存**

```bash
sudo make install
sudo ldconfig
```

预期：安装完成且无错误。

- [ ] **步骤 5：让 Python 发现绑定**

```bash
echo '/usr/local/lib/python3.6/pyrealsense2' | \
  sudo tee /usr/local/lib/python3.6/dist-packages/pyrealsense2.pth
```

预期：Python 搜索路径配置创建成功。

### 任务 4：验证 librealsense 与 D435

- [ ] **步骤 1：验证 Python 绑定**

```bash
python3 -c "import pyrealsense2 as rs; print('pyrealsense2:', rs.__file__)"
```

预期：输出 `pyrealsense2` 动态库路径。

- [ ] **步骤 2：验证设备枚举**

```bash
rs-enumerate-devices
```

预期：输出 D435 名称、序列号和支持的流配置。

- [ ] **步骤 3：验证基础深度采集**

```bash
rs-depth
```

预期：终端持续输出深度数据；按 `Ctrl+C` 结束。

### 任务 5：以测试驱动方式实现相机封装

**文件：**

- 新建：`vision/tests/test_realsense_camera.py`
- 修改：`vision/camera/realsense_camera.py`

- [ ] **步骤 1：编写失败测试**

测试应使用假的 `pyrealsense2` 对象验证：

- 默认配置为 640x480、30 FPS。
- `start()` 配置彩色流与深度流。
- `capture_aligned()` 返回彩色帧、对齐深度帧与深度内参。
- `stop()` 停止管线。

- [ ] **步骤 2：运行测试并确认失败**

```powershell
py -3 -m pytest vision/tests/test_realsense_camera.py -v
```

预期：因相机封装尚未实现而失败。

- [ ] **步骤 3：实现最小相机封装**

实现以下接口：

```python
class RealSenseCamera:
    def start(self) -> None: ...
    def capture_aligned(self): ...
    def stop(self) -> None: ...
```

`pyrealsense2` 必须延迟导入，使 PC 端测试无需安装真实硬件依赖。

- [ ] **步骤 4：运行测试并确认通过**

```powershell
py -3 -m pytest vision/tests/test_realsense_camera.py -v
```

预期：所有相机封装测试通过。

- [ ] **步骤 5：提交相机封装**

```bash
git add vision/camera/realsense_camera.py vision/tests/test_realsense_camera.py
git commit -m "实现 D435 对齐采集封装"
```

### 任务 6：实现 Jetson 相机冒烟测试

**文件：**

- 修改：`vision/tools/check_camera.py`
- 修改：`vision/README.md`

- [ ] **步骤 1：实现命令行冒烟测试**

脚本应持续采集对齐帧，打印：

```text
帧编号、彩色图尺寸、深度图尺寸、中心点深度米数
```

脚本捕获 `KeyboardInterrupt`，并始终释放相机资源。

- [ ] **步骤 2：运行 PC 静态测试**

```powershell
py -3 -m pytest vision/tests -v
```

预期：所有不依赖真实硬件的测试通过。

- [ ] **步骤 3：部署至 Jetson 外部存储**

使用 Xshell/Xftp 将 `vision` 目录同步到：

```text
/media/jetson/1896-8302/zhufeng/vision_project
```

- [ ] **步骤 4：在 Jetson 运行冒烟测试**

```bash
cd /media/jetson/1896-8302/zhufeng/vision_project
python3 tools/check_camera.py
```

预期：连续输出合理的中心点深度，按 `Ctrl+C` 后正常退出。

### 任务 7：完成十分钟稳定性验收与文档记录

- [ ] **步骤 1：运行十分钟采集测试**

```bash
cd /media/jetson/1896-8302/zhufeng/vision_project
timeout 600 python3 tools/check_camera.py
```

预期：运行至 `timeout` 结束，无 Python 异常和相机断连。

- [ ] **步骤 2：记录环境与验收结果**

将 OpenCV 版本、librealsense 版本、D435 序列号、十分钟测试结果和遇到的问题，
以中文写入主项目 Markdown 文档的修订记录。

- [ ] **步骤 3：运行最终检查**

```bash
python3 -c "import cv2, pyrealsense2; print(cv2.__version__)"
rs-enumerate-devices
df -h /
```

预期：两个 Python 模块可导入、D435 可枚举、根文件系统剩余空间大于 4 GB。

- [ ] **步骤 4：提交文档**

```bash
git add vision/README.md "vision/jetson_nano_d435_desktop_sorting_project (5).md"
git commit -m "记录 D435 RSUSB 环境验收结果"
```

- [ ] **步骤 5：清理 eMMC 临时构建目录**

确认所有验收通过后执行：

```bash
rm -rf -- /home/jetson/build/librealsense
df -h /
```

预期：临时源码和构建产物被删除，根文件系统剩余空间恢复并大于 4 GB。

## 计划自审

- 已覆盖设计中的 OpenCV、RSUSB、Python 绑定、设备枚举、RGB-D 对齐和十分钟稳定性验收。
- 内核补丁、CUDA、TensorRT、PyTorch 和识别算法均明确延后。
- 所有新增或修改的 Markdown 文档均使用中文。
- eMMC 仅保存系统、运行库和关键配置；项目工作文件全部保存在外部存储。
- 外部存储为 `vfat`，不用于 Git 仓库或动态库构建；临时构建在 eMMC 完成后清理。
