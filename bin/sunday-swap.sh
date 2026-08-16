#!/usr/bin/env bash
# Sunday late-swap check. Intended cron/n8n times (ET): 11:30, 15:45, 19:45.
# Usage: bin/sunday-swap.sh <season> <week> <salary_csv>
set -u
cd "$(dirname "$0")/.."
[ -d .venv ] && . .venv/bin/activate
[ -f .env ] && set -a && . ./.env && set +a
SEASON=${1:?season} WEEK=${2:?week} CSV=${3:?salary csv}
PYTHONPATH=src python3 -m dfs.cli swap \
  --csv "$CSV" --season "$SEASON" --week "$WEEK" \
  --log-db data/results.db --export "data/lineups/swap-w${WEEK}.csv" \
  2>&1 | tee "data/lineups/swap-w${WEEK}-$(date +%H%M).log"
