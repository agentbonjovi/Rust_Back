#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="ML Optimizer.app"
APP_DIR="$ROOT_DIR/$APP_NAME"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
MODULE_CACHE_DIR="/tmp/ml_optimizer_swift_module_cache"

mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"
mkdir -p "$MODULE_CACHE_DIR"

cat > "$CONTENTS_DIR/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>MLOptimizer</string>
  <key>CFBundleIdentifier</key>
  <string>local.r2d2.mloptimizer</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>ML Optimizer</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>LSMinimumSystemVersion</key>
  <string>13.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
PLIST

xcrun swiftc \
  -O \
  -module-cache-path "$MODULE_CACHE_DIR" \
  -framework Cocoa \
  -framework WebKit \
  "$ROOT_DIR/mac_app/MLOptimizerApp.swift" \
  -o "$MACOS_DIR/MLOptimizer"

echo "Собрано приложение: $APP_DIR"
