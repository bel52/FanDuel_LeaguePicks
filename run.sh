#!/usr/bin/env bash
# Convenience wrapper: ./run.sh build --csv ... 
cd "$(dirname "$0")"
[ -d .venv ] && . .venv/bin/activate
[ -f .env ] && set -a && . ./.env && set +a
PYTHONPATH=src python3 -m dfs.cli "$@"
