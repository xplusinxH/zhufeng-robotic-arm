# Jetson Nano D435 RSUSB Environment Design

## Goal

Prepare the Jetson Nano to capture aligned RGB-D frames from one Intel
RealSense D435 without changing the Jetson kernel.

## Confirmed Hardware And Platform

- Jetson Nano B01 running L4T R32.7.4 / JetPack 4.6.4.
- Python 3.6.9.
- Intel RealSense D435 visible over USB 3.0 as device `8086:0b07`.
- Root filesystem has about 8 GB free.
- External 30 GB storage is mounted at `/media/jetson/1896-8302`.
- Serial devices `/dev/ttyTHS1` and `/dev/ttyTHS2` are visible.

## Approach

Use the librealsense RSUSB backend. This avoids kernel patches and is the
lowest-risk route for the existing JetPack 4.6.4 installation.

Install lightweight system packages and OpenCV through Ubuntu packages.
Clone and build librealsense on the external storage with:

- RSUSB backend enabled.
- Python bindings enabled for the system Python 3.6.
- Examples and graphical tools disabled unless needed for diagnosis.
- A conservative parallel build setting suitable for the Nano's memory.

Do not install or upgrade CUDA, TensorRT, PyTorch, the Linux kernel, or the
JetPack release during this phase.

## Storage Layout

- Source and build directory:
  `/media/jetson/1896-8302/src/librealsense`
- Vision project deployment directory:
  `/media/jetson/1896-8302/vision_project`
- System libraries and Python bindings:
  installed through the normal system prefix after a successful build

## Execution Stages

1. Refresh package metadata and install basic build tools plus system OpenCV.
2. Verify Python can import OpenCV.
3. Clone a compatible librealsense release and configure the RSUSB build.
4. Build and install librealsense plus Python bindings.
5. Verify device enumeration and `pyrealsense2` import.
6. Implement and run an RGB-D alignment and depth-reading smoke test.

Each stage must pass before proceeding to the next stage.

## Error Handling

- Stop if package installation reports dependency conflicts.
- Stop if root free space drops below 3 GB.
- Keep compilation artifacts on external storage.
- Use a low parallel build count to reduce out-of-memory risk.
- Do not apply kernel patches as a fallback.

## Acceptance Criteria

- `python3 -c "import cv2; print(cv2.__version__)"` succeeds.
- `rs-enumerate-devices` detects the D435.
- `python3 -c "import pyrealsense2"` succeeds.
- A test program captures color and depth streams at 640x480.
- Depth is aligned to color.
- A selected color pixel returns a plausible depth value.
- The capture test runs for at least 10 minutes without crashing.

## Deferred Work

- CUDA, TensorRT, and PyTorch setup.
- Object detection and classification.
- Table-plane calibration and segmentation.
- Robot-arm motion and serial integration.
