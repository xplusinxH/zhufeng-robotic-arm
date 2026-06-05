# Vision Environment Design

## Goal

Set up the first project foundation for the Jetson Nano + RealSense D435 vision system while keeping the PC and Jetson responsibilities separate.

## Confirmed Boundary

The PC side is only for editing, static checks, lightweight tests, and SSH-based development. It must not be treated as the real runtime for camera, CUDA, TensorRT, or Jetson-specific libraries.

The Jetson Nano is the real runtime target. All hardware-facing dependencies must match Jetson Nano B01 with L4T R32.7.4 / JetPack 4.6.4. RealSense, CUDA, cuDNN, TensorRT, and PyTorch decisions must be checked on the Jetson before installation.

## Architecture

The repository will contain the project skeleton described in the main markdown document, plus environment files split by platform:

- `requirements-dev.txt` for the PC editing and test environment.
- `requirements-jetson.txt` for Jetson Python packages that are reasonable to install with pip.
- `scripts/jetson_check_env.sh` to inspect Jetson OS, storage, CUDA, OpenCV, RealSense, and serial readiness.
- `scripts/jetson_setup_base.sh` to install only lightweight system tools and Python support packages on Jetson.
- `README.md` to explain how to create the PC virtual environment and how to run Jetson checks over SSH.

## Dependency Strategy

The PC virtual environment installs only portable development dependencies: `numpy`, `opencv-python`, `pyserial`, `PyYAML`, `pytest`, and `ruff`.

Jetson hardware dependencies are not installed on the PC. Jetson setup is handled by scripts that must be run on the Jetson after connecting by SSH. CUDA and TensorRT are not force-installed in this first step because the project document states they are not required for phase 1 and must match JetPack.

## Verification

PC verification:

- `.venv` exists.
- `python -m pip` works inside `.venv`.
- Dev requirements install successfully or the failure is recorded.
- `python -m pytest` runs.

Jetson verification:

- Run `scripts/jetson_check_env.sh` on Jetson.
- Confirm `/etc/nv_tegra_release`, storage, Python, OpenCV, `rs-enumerate-devices`, `pyrealsense2`, and serial devices.

## Main Document Update Rule

The existing markdown file remains the master project document. This setup adds a revision entry documenting the environment split and first repository scaffold.
