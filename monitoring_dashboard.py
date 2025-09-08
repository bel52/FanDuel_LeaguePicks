"""
DFS Monitoring and Analytics Dashboard
Real-time monitoring, performance tracking, and success analytics.
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import psutil
import aiofiles

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for system monitoring."""
    timestamp: datetime
    endpoint: str
    duration_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class LineupPerformance:
    """Lineup performance tracking."""
    lineup_id: str
    contest_type: str
    projected_score: float
    actual_score: Optional[float]
    rank: Optional[int]
    total_entries: Optional[int]
    profit_loss: Optional[float]
    timestamp: datetime

@dataclass
class OptimizationStats:
    """Optimization algorithm performance stats."""
    algorithm: str
    player_pool_size: int
    optimization_time_ms: float
    lineups_generated: int
    avg_projection: float
    projection_variance: float
    timestamp: datetime

class MetricsCollector:
    """Collects and aggregates system metrics."""
    
    def __init__(self, max_history=10000):
        self.performance_history = deque(maxlen=max_history)
        self.lineup_history = deque(maxlen=max_history)
        self.optimization_history = deque(maxlen=max_history)
        self.error_counts = defaultdict(int)
        self.api_call_counts = defaultdict(int)
        self.start_time = datetime.now()
        
    def record_performance(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        self.performance_history.append(metrics)
        
        if not metrics.success and metrics.error_message:
            self.error_counts[metrics.error_message] += 1
        
        self.api_call_counts[metrics.endpoint] += 1
    
    def record_lineup_performance(self, lineup_perf: LineupPerformance):
        """Record lineup performance."""
        self.lineup_history.append(lineup_perf)
    
    def record_optimization_stats(self, opt_stats: OptimizationStats):
        """Record optimization statistics."""
        self.optimization_history.append(opt_stats)
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.performance_history if m.timestamp >= cutoff]
        
        if not recent_metrics:
            return {"message": "No recent metrics available"}
        
        durations = [m.duration_ms for m in recent_metrics]
        success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
        
        return {
            "time_period_hours": hours,
            "total_requests": len(recent_metrics),
            "success_rate": round(success_rate * 100, 2),
            "avg_response_time_ms": round(statistics.mean(durations), 2),
            "p95_response_time_ms": round(statistics.quantiles(durations, n=20)[18], 2) if len(durations) > 1 else durations[0],
            "p99_response_time_ms": round(statistics.quantiles(durations, n=100)[98], 2) if len(durations) > 1 else durations[0],
            "top_errors": dict(sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "endpoint_usage": dict(sorted(self.api_call_counts.items(), key=lambda x: x[1], reverse=True))
        }
    
    def get_lineup_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get lineup performance analytics."""
        cutoff = datetime.now() - timedelta(days=days)
        recent_lineups = [lp for lp in self.lineup_history if lp.timestamp >= cutoff]
        
        if not recent_lineups:
            return {"message": "No recent lineup data available"}
        
        completed_lineups = [lp for lp in recent_lineups if lp.actual_score is not None]
        
        if not completed_lineups:
            return {"message": "No completed lineups in time period"}
        
        actual_scores = [lp.actual_score for lp in completed_lineups]
        projected_scores = [lp.projected_score for lp in completed_lineups]
        
        # Projection accuracy
        accuracy_errors = [abs(actual - projected) for actual, projected in zip(actual_scores, projected_scores)]
        
        # Profitability
        profits = [lp.profit_loss for lp in completed_lineups if lp.profit_loss is not None]
        
        analytics = {
            "time_period_days": days,
            "total_lineups": len(recent_lineups),
            "completed_lineups": len(completed_lineups),
            "avg_actual_score": round(statistics.mean(actual_scores), 2),
            "avg_projected_score": round(statistics.mean(projected_scores), 2),
            "projection_accuracy_mae": round(statistics.mean(accuracy_errors), 2),
            "score_variance": round(statistics.variance(actual_scores), 2) if len(actual_scores) > 1 else 0,
        }
        
        if profits:
            analytics.update({
                "total_profit_loss": round(sum(profits), 2),
                "avg_profit_per_lineup": round(statistics.mean(profits), 2),
                "win_rate": round(sum(1 for p in profits if p > 0) / len(profits) * 100, 2),
                "roi_percent": round(sum(profits) / len(profits) * 100, 2) if len(profits) > 0 else 0
            })
        
        return analytics
    
    def get_optimization_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get optimization performance analytics."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_opts = [opt for opt in self.optimization_history if opt.timestamp >= cutoff]
        
        if not recent_opts:
            return {"message": "No recent optimization data available"}
        
        times = [opt.optimization_time_ms for opt in recent_opts]
        projections = [opt.avg_projection for opt in recent_opts]
        
        return {
            "time_period_hours": hours,
            "total_optimizations": len(recent_opts),
            "avg_optimization_time_ms": round(statistics.mean(times), 2),
            "max_optimization_time_ms": round(max(times), 2),
            "avg_lineup_projection": round(statistics.mean(projections), 2),
            "projection_consistency": round(statistics.stdev(projections), 2) if len(projections) > 1 else 0,
            "algorithms_used": list(set(opt.algorithm for opt in recent_opts))
        }

class SystemMonitor:
    """Real-time system monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.monitoring = False
        self.system_alerts = []
        
    async def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous system monitoring."""
        self.monitoring = True
        logger.info("Starting system monitoring...")
        
        while self.monitoring:
            try:
                await self._collect_system_metrics()
                await self._check_system_health()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(interval_seconds)
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        logger.info("System monitoring stopped")
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU and Memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Log system metrics
            system_metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                endpoint="system_health",
                duration_ms=0.0,
                memory_usage_mb=memory.used / 1024 / 1024,
                cpu_usage_percent=cpu_percent,
                success=True
            )
            
            self.metrics.record_performance(system_metrics)
            
            # Check for alerts
            if cpu_percent > 80:
                self.system_alerts.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > 85:
                self.system_alerts.append(f"High memory usage: {memory.percent:.1f}%")
            
            if disk.percent > 90:
                self.system_alerts.append(f"High disk usage: {disk.percent:.1f}%")
                
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    async def _check_system_health(self):
        """Check overall system health."""
        try:
            # Check recent error rates
            recent_summary = self.metrics.get_performance_summary(hours=1)
            
            if isinstance(recent_summary, dict) and "success_rate" in recent_summary:
                success_rate = recent_summary["success_rate"]
                
                if success_rate < 95:
                    self.system_alerts.append(f"Low success rate: {success_rate:.1f}%")
                
                avg_response = recent_summary.get("avg_response_time_ms", 0)
                if avg_response > 5000:  # 5 second threshold
                    self.system_alerts.append(f"Slow response times: {avg_response:.0f}ms")
            
            # Trim old alerts (keep last 50)
            if len(self.system_alerts) > 50:
                self.system_alerts = self.system_alerts[-50:]
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")

class DashboardAPI:
    """API endpoints for dashboard data."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.monitor = SystemMonitor(metrics_collector)
        
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "system_overview": self._get_system_overview(),
            "performance_summary": self.metrics.get_performance_summary(hours=24),
            "lineup_analytics": self.metrics.get_lineup_analytics(days=7),
            "optimization_analytics": self.metrics.get_optimization_analytics(hours=24),
            "recent_alerts": self.monitor.system_alerts[-10:],  # Last 10 alerts
            "uptime_seconds": int((datetime.now() - self.metrics.start_time).total_seconds())
        }
    
    def _get_system_overview(self) -> Dict[str, Any]:
        """Get high-level system overview."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "status": "operational",
                "cpu_usage_percent": round(cpu_percent, 1),
                "memory_usage_percent": round(memory.percent, 1),
                "disk_usage_percent": round(disk.percent, 1),
                "active_processes": len(psutil.pids()),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get system overview: {e}")
            return {"status": "error", "message": str(e)}
    
    async def export_metrics(self, filepath: str, format: str = "json"):
        """Export metrics to file."""
        try:
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "performance_metrics": [asdict(m) for m in list(self.metrics.performance_history)],
                "lineup_performance": [asdict(lp) for lp in list(self.metrics.lineup_history)],
                "optimization_stats": [asdict(opt) for opt in list(self.metrics.optimization_history)]
            }
            
            # Convert datetime objects to strings for JSON serialization
            def datetime_converter(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object {obj} is not JSON serializable")
            
            async with aiofiles.open(filepath, 'w') as f:
                if format.lower() == "json":
                    await f.write(json.dumps(data, indent=2, default=datetime_converter))
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Metrics exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False

# Performance monitoring decorator
def monitor_performance(metrics_collector: MetricsCollector, endpoint: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            success = True
            error_message = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                metrics = PerformanceMetrics(
                    timestamp=datetime.now(),
                    endpoint=endpoint,
                    duration_ms=(end_time - start_time) * 1000,
                    memory_usage_mb=end_memory,
                    cpu_usage_percent=psutil.cpu_percent(),
                    success=success,
                    error_message=error_message
                )
                
                metrics_collector.record_performance(metrics)
        
        return wrapper
    return decorator

# Example usage and testing
async def test_monitoring_system():
    """Test the monitoring system."""
    print("Testing DFS Monitoring System...")
    
    # Initialize components
    metrics_collector = MetricsCollector()
    dashboard = DashboardAPI(metrics_collector)
    
    # Simulate some metrics
    for i in range(100):
        metrics = PerformanceMetrics(
            timestamp=datetime.now() - timedelta(minutes=i),
            endpoint=f"test_endpoint_{i % 5}",
            duration_ms=50 + (i * 10) % 500,
            memory_usage_mb=100 + i,
            cpu_usage_percent=10 + (i * 2) % 40,
            success=i % 10 != 0,  # 90% success rate
            error_message="Test error" if i % 10 == 0 else None
        )
        metrics_collector.record_performance(metrics)
    
    # Get dashboard data
    dashboard_data = await dashboard.get_dashboard_data()
    
    print("Dashboard Data:")
    print(f"  System Status: {dashboard_data['system_overview']['status']}")
    print(f"  Total Requests: {dashboard_data['performance_summary']['total_requests']}")
    print(f"  Success Rate: {dashboard_data['performance_summary']['success_rate']}%")
    print(f"  Avg Response Time: {dashboard_data['performance_summary']['avg_response_time_ms']}ms")
    
    # Test export
    export_success = await dashboard.export_metrics("test_metrics.json")
    print(f"  Export Success: {export_success}")
    
    print("Monitoring system test completed!")

if __name__ == "__main__":
    asyncio.run(test_monitoring_system())
