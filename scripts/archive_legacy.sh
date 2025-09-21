#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATE="$(date +%F)"
ARCHIVE_DIR="archive/${DATE}"
LIST="${ROOT}/archive_files.txt"

if [[ ! -f "$LIST" ]]; then
  echo "Missing archive_files.txt. Run your analyzer first."
  exit 1
fi

# If list is empty, bail out gracefully
if [[ ! -s "$LIST" ]]; then
  echo "Nothing to archive (archive_files.txt is empty)."
  exit 0
fi

echo "==> Safety tag"
git add -A && git commit -m "WIP: before archiving ${DATE}" || true
git tag -f "pre-clean-${DATE}"
git push origin "pre-clean-${DATE}" || true

echo "==> Create ${ARCHIVE_DIR}"
mkdir -p "${ARCHIVE_DIR}"

echo "==> Preview (first 30)"
head -n 30 "$LIST" | sed "s#^#  will move: #"

read -rp "Proceed to move ALL listed files into ${ARCHIVE_DIR}? (yes/no) " go
if [[ "${go}" != "yes" ]]; then
  echo "Aborted."
  exit 0
fi

# Move each file, keeping relative paths under archive/
while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  # Make sure parent exists in archive
  mkdir -p "${ARCHIVE_DIR}/$(dirname "$f")"
  git mv "$f" "${ARCHIVE_DIR}/$f"
done < "$LIST"

echo "==> Commit & push"
git add -A
git commit -m "Archive legacy code to ${ARCHIVE_DIR}; keep only files reachable from main.py"
git push origin dev

echo "==> Done."
echo "Rollback: git reset --hard pre-clean-${DATE} && git push --force-with-lease origin dev"
