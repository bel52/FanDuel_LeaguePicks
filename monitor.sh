# monitor.sh - System monitoring script
#!/bin/bash

echo "📊 DFS System Monitor - Real-time Status"

# Activate virtual environment
source venv/bin/activate

# Load environment variables
set -a
source .env
set +a

# Function to check system health
check_health() {
    echo "$(date): Checking system health..."
    
    # Check if server is running
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "✅ Server is running"
    else
        echo "❌ Server is not responding"
    fi
    
    # Check system resources
    echo "💾 Memory usage: $(free -h | awk 'NR==2{printf "%.1f%%\n", $3*100/$2}')"
    echo "🖥️  CPU usage: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')"
    echo "💿 Disk usage: $(df -h / | awk 'NR==2{print $5}')"
    
    # Check log for errors
    if [ -f app/logs/dfs.log ]; then
        error_count=$(tail -n 100 app/logs/dfs.log | grep -c "ERROR")
        echo "🚨 Recent errors: $error_count"
    fi
    
    echo "---"
}

# Monitor loop
case "${1:-status}" in
    "continuous")
        echo "🔄 Starting continuous monitoring (Ctrl+C to stop)..."
        while true; do
            check_health
            sleep 30
        done
        ;;
    "logs")
        echo "📋 Showing recent logs..."
        if [ -f app/logs/dfs.log ]; then
            tail -f app/logs/dfs.log
        else
            echo "No logs found. Start the server first."
        fi
        ;;
    "dashboard")
        echo "📊 Opening monitoring dashboard..."
        python3 -c "
import asyncio
import sys
sys.path.insert(0, 'app')
from monitoring_dashboard import DashboardAPI, MetricsCollector

async def show_dashboard():
    collector = MetricsCollector()
    dashboard = DashboardAPI(collector)
    data = await dashboard.get_dashboard_data()
    
    print('\\n📊 DFS System Dashboard')
    print('=' * 50)
    print(f'Status: {data[\"system_overview\"][\"status\"]}')
    print(f'Uptime: {data[\"uptime_seconds\"]} seconds')
    print(f'CPU: {data[\"system_overview\"][\"cpu_usage_percent\"]}%')
    print(f'Memory: {data[\"system_overview\"][\"memory_usage_percent\"]}%')
    print('\\nPerformance Summary (24h):')
    perf = data['performance_summary']
    if 'total_requests' in perf:
        print(f'  Total Requests: {perf[\"total_requests\"]}')
        print(f'  Success Rate: {perf[\"success_rate\"]}%')
        print(f'  Avg Response: {perf[\"avg_response_time_ms\"]}ms')

asyncio.run(show_dashboard())
"
        ;;
    *)
        check_health
        ;;
esac
