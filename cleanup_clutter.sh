#!/bin/bash

echo "🧹 Cleaning FanDuel DFS Project Directory"
echo "========================================"

# Navigate to project directory
cd /home/brett/fanduel

echo "📋 Files to be removed (review first):"

# List clutter files for review
find . -maxdepth 1 \( \
    -name "*_backup*" -o \
    -name "*_fix*" -o \
    -name "*_simple*" -o \
    -name "*_integration_*" -o \
    -name "*.backup" -o \
    -name "*.bak" -o \
    -name "*.old" -o \
    -name "check_*" -o \
    -name "quick_*" -o \
    -name "debug_*" -o \
    -name "test_*" -o \
    -name "install_*" -o \
    -name "setup_*" -o \
    -name "manual_*" -o \
    -name "create_*" -o \
    -name "consolidate_*" -o \
    -name "clean_*" -o \
    -name "fix_*" -o \
    -name "run_*" -o \
    -name "start_*" -o \
    -name "monitor*" -o \
    -name "deploy*" -o \
    -name "*.log" \
\) -type f

echo ""
read -p "Remove these files? (y/N): " confirm

if [[ $confirm == [yY] ]]; then
    echo "🗑️ Removing clutter files..."
    
    find . -maxdepth 1 \( \
        -name "*_backup*" -o \
        -name "*_fix*" -o \
        -name "*_simple*" -o \
        -name "*_integration_*" -o \
        -name "*.backup" -o \
        -name "*.bak" -o \
        -name "*.old" -o \
        -name "check_*" -o \
        -name "quick_*" -o \
        -name "debug_*" -o \
        -name "test_*" -o \
        -name "install_*" -o \
        -name "setup_*" -o \
        -name "manual_*" -o \
        -name "create_*" -o \
        -name "consolidate_*" -o \
        -name "clean_*" -o \
        -name "fix_*" -o \
        -name "run_*" -o \
        -name "start_*" -o \
        -name "monitor*" -o \
        -name "deploy*" -o \
        -name "*.log" \
    \) -type f -delete
    
    echo "✅ Clutter files removed"
else
    echo "❌ Aborted - no files removed"
fi

echo ""
echo "📁 Cleaning directories..."
echo "Removing old logs and cache..."

# Clean logs older than 7 days
find logs/ -name "*.log" -mtime +7 -delete 2>/dev/null || true

# Clean cache if exists
rm -rf cache/* 2>/dev/null || true

echo ""
echo "🔧 Organizing core files..."

# Ensure proper structure
mkdir -p data/{input,output,lineups}
mkdir -p logs

echo ""
echo "📊 Current directory structure:"
ls -la | grep -E "^-.*\.(py|sh|txt|yml|yaml|json|md)$" | head -15

echo ""
echo "🎯 Keep these core files:"
echo "  ✅ main.py - Entry point (working)"
echo "  ✅ config.py - Configuration" 
echo "  ✅ data_collector.py - Data collection"
echo "  ✅ optimizer.py - Lineup optimization"
echo "  ✅ scheduler.py - Automated scheduling"
echo "  ✅ app.py - Web interface"
echo "  ✅ fanduel_salary_scraper.py - Salary handling"
echo "  ✅ requirements.txt - Dependencies"
echo "  ✅ .env.example - Config template"
echo "  ✅ docker-compose.yml - Container setup"
echo ""
echo "🧹 Cleanup complete!"
