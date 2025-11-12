#!/usr/bin/env bash
set -euo pipefail

# === Activate environment ===
if [ ! -d ".venv" ]; then
  echo "❌ No virtual environment found. Run setup.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# === Load environment variables ===
if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs -d '\n')
  echo "🔐 Loaded .env variables."
else
  echo "⚠️  No .env file found — please create one using .env.example"
fi

# === Run the app ===
echo "🚀 Launching NDA Review Copilot..."
streamlit run app.py