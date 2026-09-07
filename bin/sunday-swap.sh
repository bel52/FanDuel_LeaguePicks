#!/usr/bin/env bash
# Sunday late-swap check. Intended cron/n8n times (ET): 11:40, 15:45, 19:45.
# 11:40, not 11:30: the league-wide inactives list is released AT 11:30 for the 1pm
# window, so a check that fires at 11:30 races the feed and can miss the very
# scratch it exists to catch. Ten minutes costs nothing against a 1:00 lock.
# Usage: bin/sunday-swap.sh <season> <week> <salary_csv>
# -e: a failed swap must exit nonzero; -o pipefail: tee must not mask it
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && . .venv/bin/activate
[ -f .env ] && set -a && . ./.env && set +a
SEASON=${1:?season} WEEK=${2:?week} CSV=${3:?salary csv}
# --require-fresh-csv 3: FanDuel marks scratched players `O` on its own player list
# when inactives post, so a re-downloaded CSV is the inactives source. A CSV older
# than three hours cannot contain today's scratches, and swapping on one is worse than
# not swapping — it looks like a verified lineup and is not.
PYTHONPATH=src python3 -m dfs.cli swap \
  --require-fresh-csv 3 \
  --csv "$CSV" --season "$SEASON" --week "$WEEK" \
  --log-db data/results.db --export "data/lineups/swap-w${WEEK}.csv" \
  2>&1 | tee "data/lineups/swap-w${WEEK}-$(date +%H%M).log"
