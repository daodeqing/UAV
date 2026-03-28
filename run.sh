#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYSTEM_DIR="$ROOT_DIR/system"

source ~/.bashrc
# 如果你使用 conda，请取消下面两行注释并改成自己的环境名
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env

cd "$SYSTEM_DIR"

python "$SYSTEM_DIR/main.py" \
  -data UNSW \
  -algo SkyGuardPFIDS \
  -nc 50 \
  -gr 100 \
  -ls 1 \
  -dev cuda \
  -did 0
