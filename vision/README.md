# Jetson Nano D435 Vision Project

This repository contains the vision-side code for the Jetson Nano + Intel RealSense D435 desktop sorting project.

The PC is used for editing, tests, and SSH-based development. The Jetson Nano is the real runtime for D435 capture, serial communication, CUDA/TensorRT checks, and deployment.

## PC Development Environment

From this folder on Windows:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -v
```

Or run the helper script:

```powershell
.\scripts\pc_setup_dev.ps1
```

Do not install Jetson-specific CUDA, TensorRT, or RealSense system packages on the PC for this project.

## Jetson Runtime Checks

Copy or sync this repository to the Jetson project directory, recommended by the master document as:

```bash
/mnt/sdcard/vision_project
```

Then run:

```bash
cd /mnt/sdcard/vision_project
bash scripts/jetson_check_env.sh
```

Only after checking the Jetson state, run the lightweight base setup if needed:

```bash
bash scripts/jetson_setup_base.sh
```

CUDA, PyTorch, TensorRT, and RealSense SDK changes must match L4T R32.7.4 / JetPack 4.6.4.
