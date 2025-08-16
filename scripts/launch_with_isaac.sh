#!/usr/bin/env bash
set -euo pipefail
: "${ISAACSIM_PATH:?Set ISAACSIM_PATH to your Isaac Sim install dir}"

# ---- Isaac Sim env 주입 전: ZSH_VERSION 가드 ----
# 방법 A) 변수 주입
export ZSH_VERSION="${ZSH_VERSION-}"

# (대안 방법 B)
# set +u
# source "$ISAACSIM_PATH/setup_conda_env.sh"
# set -u

# 방법 A를 쓴다면 아래처럼 바로 소싱
source "$ISAACSIM_PATH/setup_conda_env.sh"

# 누락 방지: world_path 기본값 주입
DEFAULT_WORLD_PATH="/home/user/Desktop/dofbot_rl.usd"
if [[ "$*" != *"--world_path"* ]]; then
  set -- --world_path "$DEFAULT_WORLD_PATH" "$@"
fi

echo "[LAUNCH] argv: $@"

exec python scripts/train_lighting_task.py "$@"
