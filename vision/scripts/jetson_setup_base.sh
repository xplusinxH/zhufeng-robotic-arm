#!/usr/bin/env bash
set -euo pipefail

echo "Installing lightweight Jetson development tools."
sudo apt update
sudo apt install -y git vim nano tmux htop python3-pip python3-venv v4l-utils usbutils

echo "Installing portable Python packages from requirements-jetson.txt."
python3 -m pip install --user --upgrade pip
python3 -m pip install --user -r requirements-jetson.txt

echo "Base setup complete. Run scripts/jetson_check_env.sh next."
