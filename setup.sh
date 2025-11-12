#!/usr/bin/env bash
set -euo pipefail

# === Detect Python 3.12 interpreter ===
PYTHON_BIN=""
for candidate in python3.12 python3 py; do
  if command -v $candidate &>/dev/null; then
    version=$($candidate --version 2>&1)
    if [[ $version == *"3.12"* ]]; then
      PYTHON_BIN=$candidate
      break
    fi
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo "❌ Python 3.12 not found. Please install Python 3.12.9 and ensure it's on PATH."
  echo "   macOS: brew install python@3.12 && brew link python@3.12"
  echo "   Linux: sudo apt install python3.12 python3.12-venv -y"
  exit 1
fi

echo "✅ Using $($PYTHON_BIN --version)"

# === Create & activate virtual environment ===
if [ -d ".venv" ]; then
  echo "🔁 Virtual environment already exists — reusing it."
else
  echo "📦 Creating virtual environment (.venv)..."
  $PYTHON_BIN -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# === Install requirements ===
echo "📦 Upgrading pip and installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete."
echo "▶️ To start the app, run:  bash run.sh"