#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

append_ignore() { grep -qxF "$1" .gitignore || printf '%s\n' "$1" >> .gitignore; }

append_ignore ''
append_ignore '# Local secrets'
append_ignore '.env'
append_ignore '.env.*'
append_ignore '.auth/'

append_ignore ''
append_ignore '# Virtual envs & caches'
append_ignore 'venv/'
append_ignore '.venv/'
append_ignore '__pycache__/'
append_ignore '.pytest_cache/'
append_ignore '.mypy_cache/'

append_ignore ''
append_ignore '# Editor & OS'
append_ignore '.idea/'
append_ignore '.vscode/'
append_ignore '.DS_Store'

append_ignore ''
append_ignore '# Logs & temp'
append_ignore 'logs/'
append_ignore 'cache/'
append_ignore '*.pyc'

append_ignore ''
append_ignore '# Archived/local-only artifacts'
append_ignore '*.bak'
append_ignore '*.bak.*'
append_ignore '*.old'
append_ignore '*.old.*'

git add .gitignore
git commit -m "Tighten .gitignore (secrets/auth/cache/venv/logs/bak/old)" || true

# Untrack if they were added previously (keeps files on disk)
git rm --cached -r .env .env.* .auth cache logs venv .venv __pycache__ .pytest_cache .mypy_cache .idea .vscode 2>/dev/null || true
# shellcheck disable=SC2046
git rm --cached -r $(git ls-files '*.pyc' '*.bak' '*.bak.*' '*.old' '*.old.*' 2>/dev/null) 2>/dev/null || true

git commit -m "Stop tracking local-only files per hardened .gitignore" || true
git push origin dev
