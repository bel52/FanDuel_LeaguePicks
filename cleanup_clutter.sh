#!/bin/bash

echo "🧹 Aggressive Cleanup FanDuel DFS Project Directory"
echo "================================================="

# Navigate to project directory
cd /home/brett/fanduel

echo "📋 Files and directories to be removed:"

# List all clutter files
echo "Files:"
ls -1 | grep -E "(=0\.|\.backup|\.broken|\.corrupted|ai_analyzer\.py|fanduel_scraper\.py|lineup_validator\.py|lineups_.*\.csv|Makefile|setup\.sh|start\.sh)" || echo "No matching files"

echo ""
echo "Directories:"
ls -1d */ | grep -E "(app/|archive/|\.auth/|cache/|scripts/|src/|tests/|tools/|static/)" || echo "No matching directories"

echo ""
read -p "🗑️ Remove these files and directories? (y/N): " confirm

if [[ $confirm == [yY] ]]; then
    echo "Removing files..."
    
    # Remove specific problematic files
    rm -f "=0.104.1" "=0.24.0" "=0.7.0" "=1.0.0"
    rm -f api.py.backup2
    rm -f ai_analyzer.py
    rm -f archive_files.txt
    rm -f config.json
    rm -f fanduel_scraper.py
    rm -f FILE_STRUCTURE.md
    rm -f keep_files.txt
    rm -f lineups_20250921_200329.csv
    rm -f lineup_validator.py
    rm -f Makefile
    rm -f optimizer.py.backup2
    rm -f optimizer.py.broken
    rm -f optimizer.py.corrupted
    rm -f setup.sh
    rm -f start.sh
    
    # Remove unnecessary directories
    rm -rf app/
    rm -rf archive/
    rm -rf .auth/
    rm -rf cache/
    rm -rf scripts/
    rm -rf src/
    rm -rf static/
    rm -rf tests/
    rm -rf tools/
    rm -rf __pycache__/
    
    echo "✅ Clutter removed"
else
    echo "❌ Aborted - no files removed"
    exit 0
fi

echo ""
echo "🔧 Organizing remaining files..."

# Clean logs older than 7 days
find logs/ -name "*.log" -mtime +7 -delete 2>/dev/null || true

echo ""
echo "📊 Final directory structure:"
ls -la

echo ""
echo "✅ Core files remaining:"
for file in main.py config.py data_collector.py optimizer.py scheduler.py app.py fanduel_salary_scraper.py requirements.txt .env.example docker-compose.yml Dockerfile README.md .gitignore; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (missing)"
    fi
done

echo ""
echo "📁 Data directories:"
ls -la data/ 2>/dev/null || echo "No data directory"

echo ""
echo "🧹 Cleanup complete!"
echo "🎯 You now have a clean, focused project structure"
