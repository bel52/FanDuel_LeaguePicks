#!/usr/bin/env bash
set -euo pipefail

BASE="http://localhost:8010"

echo "Generating optimal lineup..."
echo ""
curl -s "$BASE/optimize/text?game_type=gpp" || {
  echo "Server not responding. Start it with: ./cli.py"
  exit 1
}
echo ""
