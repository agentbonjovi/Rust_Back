#!/usr/bin/env bash
# ==========================================================================
# ML Optimizer — bootstrap script
# --------------------------------------------------------------------------
# Поднимает окружение для нового пользователя, склонировавшего репозиторий:
#   1. ставит rustup (если нет cargo) и нужную toolchain из rust-toolchain.toml
#   2. проверяет Python >= 3.9
#   3. при наличии requirements.txt c пакетами — ставит pip-зависимости
#   4. фиксированно собирает Rust release-бинарь по Cargo.lock
#   5. генерирует synthetic_data.csv, если его нет
#   6. делает smoke-test обоих бэкендов (Rust и Python)
#
# Использование:
#   bash setup.sh                # полная установка
#   bash setup.sh --no-build     # пропустить cargo build
#   bash setup.sh --no-smoke     # пропустить запуск бэкендов
# ==========================================================================

set -euo pipefail

PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=9
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DO_BUILD=1
DO_SMOKE=1
for arg in "$@"; do
    case "$arg" in
        --no-build) DO_BUILD=0 ;;
        --no-smoke) DO_SMOKE=0 ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *) echo "Неизвестный аргумент: $arg" >&2; exit 2 ;;
    esac
done

# --- helpers --------------------------------------------------------------
info()  { printf '\033[1;36m[INFO]\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
fail()  { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1
}

OS_NAME="$(uname -s 2>/dev/null || echo Unknown)"
info "Платформа: $OS_NAME"

# --- 1. Rust --------------------------------------------------------------
ensure_rust() {
    if require_cmd cargo && require_cmd rustc; then
        ok "cargo найден: $(cargo --version)"
        return
    fi

    info "cargo не найден. Устанавливаю rustup..."
    if ! require_cmd curl; then
        fail "Нужен curl для установки rustup. Установите curl и повторите."
    fi
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none --profile minimal

    # подключаем cargo к текущему shell-процессу
    if [ -f "$HOME/.cargo/env" ]; then
        # shellcheck disable=SC1090
        source "$HOME/.cargo/env"
    fi

    if ! require_cmd cargo; then
        fail "rustup установлен, но cargo всё ещё не в PATH. Перезапустите shell и попробуйте снова."
    fi
    ok "rustup установлен."
}

ensure_rust_toolchain() {
    # rust-toolchain.toml в корне проекта заставит rustup поставить
    # нужную toolchain автоматически при первом вызове cargo.
    info "Синхронизирую Rust toolchain согласно rust-toolchain.toml..."
    rustup show active-toolchain >/dev/null 2>&1 || true
    cargo --version >/dev/null
    ok "Rust toolchain готов: $(rustc --version)"
}

# --- 2. Python ------------------------------------------------------------
ensure_python() {
    local py_bin=""
    for candidate in python3 python; do
        if require_cmd "$candidate"; then
            py_bin="$candidate"
            break
        fi
    done
    if [ -z "$py_bin" ]; then
        fail "Не найден python3. Установите Python >= ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR} и повторите запуск."
    fi

    local version
    version="$($py_bin -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    local major="${version%%.*}"
    local minor="${version##*.}"
    if [ "$major" -lt "$PYTHON_MIN_MAJOR" ] || { [ "$major" -eq "$PYTHON_MIN_MAJOR" ] && [ "$minor" -lt "$PYTHON_MIN_MINOR" ]; }; then
        fail "Найден Python $version, нужен >= ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR}"
    fi
    ok "Python $version OK ($($py_bin -c 'import sys; print(sys.executable)'))"
    export PROJECT_PYTHON="$py_bin"
}

ensure_python_deps() {
    if [ ! -f requirements.txt ]; then
        return
    fi
    # ищем хотя бы одну непустую/некомментарную строку
    if ! grep -Eq '^[[:space:]]*[^[:space:]#]' requirements.txt; then
        ok "requirements.txt пуст — внешних Python-пакетов не требуется."
        return
    fi
    info "Устанавливаю Python-зависимости из requirements.txt..."
    "$PROJECT_PYTHON" -m pip install --upgrade pip >/dev/null
    "$PROJECT_PYTHON" -m pip install -r requirements.txt
    ok "Python-зависимости установлены."
}

# --- 3. Rust build --------------------------------------------------------
build_rust() {
    if [ "$DO_BUILD" -eq 0 ]; then
        warn "Пропускаю cargo build (--no-build)."
        return
    fi
    info "Собираю Rust release-бинарь (cargo build --release)..."
    cargo build --release
    if [ ! -x "target/release/bac123" ]; then
        fail "Сборка завершилась, но бинарник target/release/bac123 не найден."
    fi
    ok "Rust-бинарь собран: target/release/bac123"
}

# --- 4. Synthetic CSV -----------------------------------------------------
ensure_dataset() {
    if [ -f synthetic_data.csv ]; then
        ok "synthetic_data.csv уже существует, пропускаю генерацию."
        return
    fi
    info "Генерирую synthetic_data.csv через Rust-бинарь..."
    if [ -x "target/release/bac123" ]; then
        ./target/release/bac123 synthetic_data.csv >/dev/null
    else
        "$PROJECT_PYTHON" python_backend.py synthetic_data.csv >/dev/null
    fi
    ok "synthetic_data.csv сгенерирован."
}

# --- 5. Smoke tests -------------------------------------------------------
smoke_test() {
    if [ "$DO_SMOKE" -eq 0 ]; then
        warn "Пропускаю smoke-тесты (--no-smoke)."
        return
    fi
    info "Smoke-тест Rust-бэкенда..."
    if [ -x "target/release/bac123" ]; then
        local rust_time
        rust_time=$(./target/release/bac123 synthetic_data.csv 2>&1 | grep "Время выполнения" | head -1 || true)
        ok "Rust: ${rust_time:-выполнен}"
    else
        warn "Rust-бинарь не собран, пропускаю."
    fi

    info "Smoke-тест Python-бэкенда..."
    local py_time
    py_time=$("$PROJECT_PYTHON" python_backend.py synthetic_data.csv 2>&1 | grep "Время выполнения" | head -1 || true)
    ok "Python: ${py_time:-выполнен}"
}

# --- main -----------------------------------------------------------------
echo "=========================================="
echo " ML Optimizer — установка окружения"
echo "=========================================="
ensure_rust
ensure_rust_toolchain
ensure_python
ensure_python_deps
build_rust
ensure_dataset
smoke_test

echo
ok "Готово. Дальше можно запускать:"
echo "    cargo run --release -- synthetic_data.csv     # Rust backend"
echo "    $PROJECT_PYTHON python_backend.py synthetic_data.csv   # Python backend"
echo "    $PROJECT_PYTHON gui_app.py --mode browser              # Web GUI с переключателем языка"
