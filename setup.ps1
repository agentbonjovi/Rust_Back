# ============================================================
# ML Optimizer - bootstrap (Windows / PowerShell)
# Аналог setup.sh для Windows. Ставит rustup, проверяет Python,
# собирает Rust release-бинарь и делает smoke-тест обоих бэкендов.
#
# Запуск:
#     powershell -ExecutionPolicy Bypass -File setup.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ProjectRoot

function Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "[ OK ] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red; exit 1 }

# --- Rust ---
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Info "cargo не найден. Устанавливаю rustup..."
    $rustupInit = Join-Path $env:TEMP "rustup-init.exe"
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $rustupInit
    & $rustupInit -y --default-toolchain none --profile minimal
    $env:Path += ";$env:USERPROFILE\.cargo\bin"
}
Ok ("cargo: " + (cargo --version))

Info "Синхронизирую Rust toolchain согласно rust-toolchain.toml..."
cargo --version | Out-Null
Ok ("rustc: " + (rustc --version))

# --- Python ---
# Перебираем кандидатов: свежие версии сначала, потом общие имена.
$candidates = @("python3.13", "python3.12", "python3.11", "python3.10", "python3.9", "python3", "python")
$python = $null
$pickedVersion = $null
foreach ($name in $candidates) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if (-not $cmd) { continue }
    $v = & $cmd.Source -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    if (-not $v) { continue }
    $parts = $v.Split(".")
    if ([int]$parts[0] -gt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 9)) {
        $python = $cmd
        $pickedVersion = $v
        break
    }
}
if (-not $python) { Fail "Не найден Python >= 3.9. Установите его с python.org" }
Ok "Python $pickedVersion OK ($($python.Source))"

# Smoke-проверка: импортируется ли gui_app под этим Python (на 3.13
# удалённый модуль cgi или distutils выскочит здесь, ещё до запуска).
$importErr = & $python.Source -c "import gui_app" 2>&1
if ($LASTEXITCODE -ne 0) {
    Warn "gui_app.py не импортируется под $($python.Source):"
    Write-Host $importErr -ForegroundColor Yellow
    Fail "Исправьте импорт-ошибку и повторите setup.ps1."
}
Ok "gui_app.py импортируется без ошибок."

# --- Python deps ---
if (Test-Path requirements.txt) {
    $hasDeps = Select-String -Path requirements.txt -Pattern '^\s*[^\s#]' -Quiet
    if ($hasDeps) {
        Info "Устанавливаю Python-зависимости..."
        & $python.Source -m pip install --upgrade pip | Out-Null
        & $python.Source -m pip install -r requirements.txt
        Ok "Python-зависимости установлены."
    } else {
        Ok "requirements.txt пуст — внешние Python-пакеты не требуются."
    }
}

# --- Build ---
Info "Собираю Rust release-бинарь..."
cargo build --release
Ok "Rust-бинарь собран."

# --- Dataset ---
if (-not (Test-Path synthetic_data.csv)) {
    Info "Генерирую synthetic_data.csv..."
    & ".\target\release\bac123.exe" synthetic_data.csv | Out-Null
    Ok "synthetic_data.csv сгенерирован."
}

# --- Smoke ---
Info "Smoke-тест Rust..."
& ".\target\release\bac123.exe" synthetic_data.csv | Select-String "Время выполнения"
Info "Smoke-тест Python..."
& $python.Source python_backend.py synthetic_data.csv | Select-String "Время выполнения"

Ok "Готово. Запуск:"
Write-Host "    cargo run --release -- synthetic_data.csv"
Write-Host "    $($python.Source) python_backend.py synthetic_data.csv"
Write-Host "    $($python.Source) gui_app.py --mode browser"
