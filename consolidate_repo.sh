#!/usr/bin/env bash
set -euo pipefail

SRC="${HOME}/FanDuel_LeaguePicks"
DST="/home/brett/fanduel"
TS="$(date +%F_%H%M%S)"

echo "==> Validating paths"
[ -d "$SRC" ] || { echo "Source not found: $SRC"; exit 1; }
[ -d "$DST" ] || { echo "Destination not found: $DST"; exit 1; }

echo "==> Backing up current working repo: ${DST}.bak_${TS}"
cp -a "$DST" "${DST}.bak_${TS}"

echo "==> Ensuring Git repo and dev branch are ready"
cd "$DST"
git rev-parse --is-inside-work-tree >/dev/null
git fetch origin
git add -A && git commit -m "WIP: save local edits before consolidation ${TS}" || true
git checkout dev || git checkout -b dev origin/dev
git pull --rebase origin dev || true

echo "==> Dry run: show what will sync from $SRC -> $DST"
rsync -avhn \
  --include='.env' \
  --include='.env.*' \
  --exclude='.git/' \
  --exclude='venv/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='.idea/' \
  --exclude='.vscode/' \
  --exclude='*.bak' \
  --exclude='*.old' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='logs/' \
  "$SRC"/ "$DST"/

read -rp "Proceed with REAL sync? (yes/no) " go
if [[ "${go}" != "yes" ]]; then
  echo "Aborted by user."
  exit 0
fi

echo "==> Real sync now..."
rsync -avh --progress \
  --include='.env' \
  --include='.env.*' \
  --exclude='.git/' \
  --exclude='venv/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='.idea/' \
  --exclude='.vscode/' \
  --exclude='*.bak' \
  --exclude='*.old' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='logs/' \
  "$SRC"/ "$DST"/

echo "==> Hardening .gitignore"
append_ignore() { grep -qxF "$1" .gitignore || printf '%s\n' "$1" >> .gitignore; }

append_ignore ''
append_ignore '# Local secrets'
append_ignore '.env'
append_ignore '.env.*'

append_ignore ''
append_ignore '# Virtual envs & caches'
append_ignore 'venv/'
append_ignore '.venv/'
append_ignore '__pycache__/'
append_ignore '.pytest_cache/'
append_ignore '.mypy_cache/'

append_ignore ''
append_ignore '# Editors & OS'
append_ignore '.idea/'
append_ignore '.vscode/'
append_ignore '.DS_Store'

append_ignore ''
append_ignore '# Logs & temp'
append_ignore 'logs/'
append_ignore '*.pyc'

append_ignore ''
append_ignore '# Archived/local-only artifacts'
append_ignore '*.bak'
append_ignore '*.old'

echo "==> Untracking any secrets/junk that might already be committed (files remain on disk)"
git rm --cached -r .env .env.* venv .venv __pycache__ .pytest_cache .mypy_cache .idea .vscode logs 2>/dev/null || true
# shellcheck disable=SC2046
git rm --cached -r $(git ls-files '*.bak' '*.old' '*.pyc' 2>/dev/null) 2>/dev/null || true

echo "==> Staging changes"
git add -A
git status

echo "==> Commit"
git commit -m "Consolidate to /home/brett/fanduel; include .env & backups locally; exclude *.bak/*.old/venv/caches/logs; tighten .gitignore (${TS})" || true

echo "==> Push to origin/dev"
git push origin dev

echo "==> Optional: smoke test (requires venv + deps). Skip with Ctrl+C now if you don't want this."
read -rp "Run smoke test now? (yes/no) " test_go
if [[ "${test_go}" == "yes" ]]; then
  echo "==> Ensuring venv & deps"
  python3 -m venv venv || true
  # shellcheck disable=SC1091
  source venv/bin/activate
  pip install -U pip
  if [ -f requirements.txt ]; then
    pip install -r requirements.txt
  fi
  echo "==> One-lineup test"
  python main.py --type balanced --lineups 1 || true
fi

echo "==> FINAL SAFETY CHECK: remove the old directory ONLY after you confirm everything is running."
read -rp "Delete source directory ${SRC}? (type: DELETE) " del
if [[ "${del}" == "DELETE" ]]; then
  echo "==> Removing ${SRC}"
  rm -rf "${SRC}"
else
  echo "Skipped deletion. You can remove later with: rm -rf \"${SRC}\""
fi

echo "==> Done. If needed, rollback with: rm -rf ${DST}; mv ${DST}.bak_${TS} ${DST}"
