#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_PATH="$ROOT_DIR/ML Optimizer.app"

if [ ! -d "$APP_PATH" ]; then
  "$ROOT_DIR/build_mac_app.sh"
fi

open "$APP_PATH"
