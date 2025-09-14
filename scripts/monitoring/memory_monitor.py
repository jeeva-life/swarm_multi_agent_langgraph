#!/usr/bin/env python3
"""
Memory monitoring dashboard for Redis-backed short-term memory.
Provides real-time monitoring and analysis of memory usage.
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime, timedelta

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.config import config
from src.core.memory.memory_backends import RedisBackend
from src.core.memory.memory_manager import MemoryManager


class MemoryMonitor:
    """Monitor and analyze Redis memory usage."""
    
    def __init__(self):
        self.logger = logging.getLogger("memory_monitor")
        self.setup_logging()
        self.redis_backend = None
        self.memory_manager = None
    
    def setup_logging(self):
        """Setup logging for memory monitoring."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def initialize(self):
        """Initialize monitoring components."""
        try:
            self.redis_backend = RedisBackend()
            self.memory_manager = MemoryManager()
            
            # Test connection
            if not await self.redis_backend.redis.ping():
                raise Exception("Redis connection failed")
            
            self.logger.info("Memory monitor initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory monitor: {str(e)}")
            return False
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            # Get Redis memory info
            memory_info = await self.redis_backend.redis.info('memory')
            stats_info = await self.redis_backend.redis.info('stats')
            
            # Get memory manager stats
            manager_stats = await self.memory_manager.get_memory_stats()
            
            # Calculate memory usage
            used_memory = memory_info.get('used_memory', 0)
            max_memory = memory_info.get('maxmemory', 0)
            memory_usage_percent = (used_memory / max_memory * 100) if max_memory > 0 else 0
            
            # Get key statistics
            db_info = await self.redis_backend.redis.info('keyspace')
            
            stats = {
                "timestamp": datetime.now().isoformat(),
                "redis_memory": {
                    "used_memory": used_memory,
                    "used_memory_human": memory_info.get('used_memory_human', '0B'),
                    "max_memory": max_memory,
                    "max_memory_human": memory_info.get('maxmemory_human', '0B'),
                    "memory_usage_percent": round(memory_usage_percent, 2),
                    "fragmentation_ratio": memory_info.get('mem_fragmentation_ratio', 0),
                    "rss_memory": memory_info.get('used_memory_rss', 0),
                    "peak_memory": memory_info.get('used_memory_peak', 0)
                },
                "redis_stats": {
                    "total_commands_processed": stats_info.get('total_commands_processed', 0),
                    "instantaneous_ops_per_sec": stats_info.get('instantaneous_ops_per_sec', 0),
                    "keyspace_hits": stats_info.get('keyspace_hits', 0),
                    "keyspace_misses": stats_info.get('keyspace_misses', 0),
                    "expired_keys": stats_info.get('expired_keys', 0),
                    "evicted_keys": stats_info.get('evicted_keys', 0)
                },
                "keyspace": db_info,
                "memory_manager": manager_stats,
                "health_status": self._assess_health_status(used_memory, max_memory, memory_usage_percent)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting memory stats: {str(e)}")
            return {"error": str(e)}
    
    def _assess_health_status(self, used_memory: int, max_memory: int, usage_percent: float) -> Dict[str, Any]:
        """Assess memory health status."""
        status = "healthy"
        warnings = []
        
        if usage_percent > 90:
            status = "critical"
            warnings.append("Memory usage above 90%")
        elif usage_percent > 80:
            status = "warning"
            warnings.append("Memory usage above 80%")
        elif usage_percent > 70:
            status = "caution"
            warnings.append("Memory usage above 70%")
        
        if max_memory == 0:
            warnings.append("No memory limit set")
        
        return {
            "status": status,
            "warnings": warnings,
            "recommendations": self._get_recommendations(usage_percent, max_memory)
        }
    
    def _get_recommendations(self, usage_percent: float, max_memory: int) -> List[str]:
        """Get memory management recommendations."""
        recommendations = []
        
        if usage_percent > 80:
            recommendations.append("Consider increasing Redis maxmemory")
            recommendations.append("Implement more aggressive TTL policies")
            recommendations.append("Enable memory cleanup more frequently")
        
        if max_memory == 0:
            recommendations.append("Set a maxmemory limit to prevent OOM")
            recommendations.append("Configure appropriate eviction policy")
        
        if usage_percent < 50:
            recommendations.append("Memory usage is healthy")
            recommendations.append("Consider increasing TTL for better performance")
        
        return recommendations
    
    async def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation-specific statistics."""
        try:
            # Get all conversation keys
            conversation_keys = await self.redis_backend.redis.keys("stm:conversation:*")
            
            stats = {
                "total_conversations": len(conversation_keys),
                "conversation_details": []
            }
            
            for key in conversation_keys[:10]:  # Limit to first 10 for performance
                try:
                    data = await self.redis_backend.get(key.replace("stm:", ""))
                    if data:
                        conversation_data = json.loads(data)
                        if isinstance(conversation_data, list):
                            stats["conversation_details"].append({
                                "key": key,
                                "message_count": len(conversation_data),
                                "last_message": conversation_data[-1] if conversation_data else None
                            })
                except Exception as e:
                    self.logger.warning(f"Error processing conversation key {key}: {str(e)}")
                    continue
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting conversation stats: {str(e)}")
            return {"error": str(e)}
    
    async def cleanup_old_data(self, max_age_hours: int = 24) -> Dict[str, int]:
        """Clean up old data to free memory."""
        try:
            self.logger.info(f"Starting cleanup of data older than {max_age_hours} hours...")
            
            # Clean up old sessions
            cleaned_sessions = await self.memory_manager.cleanup_old_sessions(max_age_hours)
            
            # Clean up expired entries
            cleanup_stats = await self.memory_manager.cleanup_expired()
            
            total_cleaned = cleaned_sessions + cleanup_stats.get("short_term_cleaned", 0) + cleanup_stats.get("long_term_cleaned", 0)
            
            self.logger.info(f"Cleanup completed: {total_cleaned} entries removed")
            
            return {
                "cleaned_sessions": cleaned_sessions,
                "cleaned_short_term": cleanup_stats.get("short_term_cleaned", 0),
                "cleaned_long_term": cleanup_stats.get("long_term_cleaned", 0),
                "total_cleaned": total_cleaned
            }
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            return {"error": str(e)}
    
    async def monitor_continuously(self, interval_seconds: int = 30, duration_minutes: int = 10):
        """Monitor memory usage continuously."""
        try:
            self.logger.info(f"Starting continuous monitoring for {duration_minutes} minutes...")
            self.logger.info(f"Update interval: {interval_seconds} seconds")
            
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)
            
            while time.time() < end_time:
                stats = await self.get_memory_stats()
                
                # Display current status
                memory_info = stats.get("redis_memory", {})
                health = stats.get("health_status", {})
                
                print(f"\n[STATUS] Memory Status - {datetime.now().strftime('%H:%M:%S')}")
                print("-" * 50)
                print(f"Used Memory: {memory_info.get('used_memory_human', 'Unknown')}")
                print(f"Memory Usage: {memory_info.get('memory_usage_percent', 0):.1f}%")
                print(f"Health Status: {health.get('status', 'Unknown').upper()}")
                
                if health.get('warnings'):
                    print("[WARNING] Warnings:")
                    for warning in health.get('warnings', []):
                        print(f"   - {warning}")
                
                if health.get('recommendations'):
                    print("[INFO] Recommendations:")
                    for rec in health.get('recommendations', []):
                        print(f"   - {rec}")
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
            
            self.logger.info("Continuous monitoring completed")
            
        except Exception as e:
            self.logger.error(f"Error in continuous monitoring: {str(e)}")
    
    async def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory report."""
        try:
            self.logger.info("Generating comprehensive memory report...")
            
            # Get all statistics
            memory_stats = await self.get_memory_stats()
            conversation_stats = await self.get_conversation_stats()
            
            # Generate report
            report = {
                "report_timestamp": datetime.now().isoformat(),
                "memory_analysis": memory_stats,
                "conversation_analysis": conversation_stats,
                "summary": {
                    "total_memory_usage": memory_stats.get("redis_memory", {}).get("used_memory_human", "Unknown"),
                    "memory_health": memory_stats.get("health_status", {}).get("status", "Unknown"),
                    "total_conversations": conversation_stats.get("total_conversations", 0),
                    "recommendations": memory_stats.get("health_status", {}).get("recommendations", [])
                }
            }
            
            # Save report
            report_file = f"memory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Report saved to {report_file}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return {"error": str(e)}


async def main():
    """Main monitoring function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory Monitor for Redis-backed Multi-Agent System")
    parser.add_argument("--mode", choices=["stats", "monitor", "cleanup", "report"], default="stats",
                       help="Monitoring mode")
    parser.add_argument("--interval", type=int, default=30,
                       help="Monitoring interval in seconds (for monitor mode)")
    parser.add_argument("--duration", type=int, default=10,
                       help="Monitoring duration in minutes (for monitor mode)")
    parser.add_argument("--max-age", type=int, default=24,
                       help="Maximum age in hours for cleanup")
    
    args = parser.parse_args()
    
    monitor = MemoryMonitor()
    
    if not await monitor.initialize():
        print("[ERROR] Failed to initialize memory monitor")
        sys.exit(1)
    
    if args.mode == "stats":
        stats = await monitor.get_memory_stats()
        print("\n[STATS] Memory Statistics")
        print("=" * 50)
        print(json.dumps(stats, indent=2))
        
    elif args.mode == "monitor":
        await monitor.monitor_continuously(args.interval, args.duration)
        
    elif args.mode == "cleanup":
        cleanup_stats = await monitor.cleanup_old_data(args.max_age)
        print("\n[CLEANUP] Cleanup Results")
        print("=" * 50)
        print(json.dumps(cleanup_stats, indent=2))
        
    elif args.mode == "report":
        report = await monitor.generate_report()
        print("\n[REPORT] Memory Report Generated")
        print("=" * 50)
        print(f"Report saved with {len(report)} sections")


if __name__ == "__main__":
    asyncio.run(main())
