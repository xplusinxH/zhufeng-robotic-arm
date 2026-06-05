# Vision Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the initial PC development environment and Jetson-targeted setup/check files for the vision project.

**Architecture:** Keep PC dependencies portable and lightweight. Keep Jetson hardware dependencies in scripts and documentation so they are checked and installed on the Jetson itself, not on the PC.

**Tech Stack:** Python 3.9, virtualenv, OpenCV Python, NumPy, PySerial, PyYAML, pytest, ruff, Jetson L4T shell checks.

---

## File Structure

- Create `.gitignore` to exclude virtual environments, caches, logs, large data, and model weights.
- Create `README.md` with PC and Jetson setup commands.
- Create `requirements-dev.txt` for the PC development venv.
- Create `requirements-jetson.txt` for Jetson Python packages.
- Create `config.yaml` with baseline camera, depth, workspace, serial, and logging settings.
- Create package directories: `camera`, `calibration`, `perception`, `coordinate`, `communication`, `tools`, `models`, `logs`, `tests`, `scripts`.
- Create placeholder Python modules with importable package files.
- Create `scripts/jetson_check_env.sh` and `scripts/jetson_setup_base.sh`.
- Create `tests/test_project_imports.py` to verify the scaffold imports.

### Task 1: Repository Foundation

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `requirements-dev.txt`
- Create: `requirements-jetson.txt`
- Create: `config.yaml`

- [ ] **Step 1: Add environment and dependency files**

Create the files listed above with a PC/Jetson split.

- [ ] **Step 2: Verify file presence**

Run: `Test-Path README.md; Test-Path requirements-dev.txt; Test-Path requirements-jetson.txt; Test-Path config.yaml`

Expected: four `True` lines.

### Task 2: Project Package Skeleton

**Files:**
- Create: `main.py`
- Create: package `__init__.py` files under `camera`, `perception`, `coordinate`, `communication`
- Create: module stubs matching the master project document
- Create: `models/README.md`
- Create: `logs/README.md`

- [ ] **Step 1: Add importable modules**

Each module should contain a docstring and minimal functions or classes only where useful for first verification.

- [ ] **Step 2: Add import test**

Create `tests/test_project_imports.py` that imports representative modules and asserts the config file exists.

- [ ] **Step 3: Run import test**

Run: `.venv\Scripts\python.exe -m pytest tests/test_project_imports.py -v`

Expected: tests pass after the virtual environment exists.

### Task 3: Jetson Scripts

**Files:**
- Create: `scripts/jetson_check_env.sh`
- Create: `scripts/jetson_setup_base.sh`

- [ ] **Step 1: Add check script**

The check script should print OS, L4T, storage, USB camera, serial device, Python, OpenCV, RealSense CLI, `pyrealsense2`, CUDA directory, and NVIDIA package status.

- [ ] **Step 2: Add base setup script**

The setup script should install only lightweight tools and Python support packages, avoiding CUDA/PyTorch/TensorRT installation.

- [ ] **Step 3: Verify scripts are present**

Run: `Test-Path scripts/jetson_check_env.sh; Test-Path scripts/jetson_setup_base.sh`

Expected: two `True` lines.

### Task 4: PC Virtual Environment

**Files:**
- Use: `.venv`
- Use: `requirements-dev.txt`

- [ ] **Step 1: Create virtual environment**

Run: `py -3 -m venv .venv`

Expected: `.venv\Scripts\python.exe` exists.

- [ ] **Step 2: Upgrade pip**

Run: `.venv\Scripts\python.exe -m pip install --upgrade pip`

Expected: pip upgrade completes.

- [ ] **Step 3: Install PC dev requirements**

Run: `.venv\Scripts\python.exe -m pip install -r requirements-dev.txt`

Expected: dependencies install.

- [ ] **Step 4: Run verification**

Run: `.venv\Scripts\python.exe -m pytest tests/test_project_imports.py -v`

Expected: all tests pass.

## Self-Review

- Spec coverage: PC/Jetson split, project skeleton, virtual environment, Jetson checks, and documentation are covered.
- Placeholder scan: No implementation placeholders are required for this environment setup; later vision algorithms will be implemented in later phases.
- Type consistency: The first test only imports modules and reads files, so no public algorithmic API is frozen yet.
