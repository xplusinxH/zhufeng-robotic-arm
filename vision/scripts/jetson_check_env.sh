#!/usr/bin/env bash
set -u

echo "== System =="
uname -a
if [ -f /etc/nv_tegra_release ]; then
  cat /etc/nv_tegra_release
else
  echo "Missing /etc/nv_tegra_release; this does not look like Jetson L4T."
fi

echo
echo "== Storage =="
df -h
lsblk

echo
echo "== USB / Camera =="
if command -v lsusb >/dev/null 2>&1; then
  lsusb
else
  echo "lsusb is not installed."
fi

echo
echo "== Serial Devices =="
ls /dev/ttyUSB* /dev/ttyTHS* 2>/dev/null || echo "No ttyUSB/ttyTHS devices found."

echo
echo "== Python =="
python3 --version
python3 -m pip --version 2>/dev/null || echo "python3 pip is not available."

echo
echo "== OpenCV =="
python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "Python OpenCV import failed."

echo
echo "== RealSense =="
if command -v rs-enumerate-devices >/dev/null 2>&1; then
  rs-enumerate-devices
else
  echo "rs-enumerate-devices is not installed or not in PATH."
fi
python3 -c "import pyrealsense2 as rs; print(rs)" 2>/dev/null || echo "pyrealsense2 import failed."

echo
echo "== CUDA / NVIDIA Packages =="
nvcc --version 2>/dev/null || echo "nvcc is not available."
ls -l /usr/local/cuda* 2>/dev/null || echo "No /usr/local/cuda* path found."
dpkg -l | grep -E "cuda|cudnn|nvinfer|tensorrt" || echo "No matching CUDA/cuDNN/TensorRT packages listed by dpkg."
