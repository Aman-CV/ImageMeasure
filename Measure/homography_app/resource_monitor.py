"""
Resource monitoring utility for GPU and CPU tracking.
Logs resource usage to file for AWS cost analysis.
"""

import os
import psutil
import logging
import time
from datetime import datetime
from typing import Dict, Optional
import threading

logger = logging.getLogger('homography_app')

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. GPU memory tracking will be disabled.")


class ResourceMonitor:
    """Monitor CPU and GPU resource usage during video processing."""
    
    def __init__(self, task_name: str, video_id: int = None, tracking_interval: float = 2.0):
        """
        Initialize resource monitor.
        
        Args:
            task_name: Name of the processing task (e.g., 'broad_jump', 'plank')
            video_id: ID of the video being processed
            tracking_interval: Seconds between resource checks (default: 2s)
        """
        self.task_name = task_name
        self.video_id = video_id
        self.tracking_interval = tracking_interval
        
        # Metrics storage
        self.metrics = {
            'task_name': task_name,
            'video_id': video_id,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'cpu_memory_peak_mb': 0,
            'cpu_memory_avg_mb': 0,
            'gpu_memory_peak_mb': 0,
            'gpu_memory_avg_mb': 0,
            'cpu_percent_avg': 0,
            'duration_seconds': 0,
            'status': 'running',
        }
        
        # Data collection
        self.cpu_memory_samples = []
        self.gpu_memory_samples = []
        self.cpu_percent_samples = []
        
        # Process reference
        self.process = psutil.Process()
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start resource monitoring in background thread."""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"[{self.task_name}] Resource monitoring started (video_id={self.video_id})")
        
    def stop(self):
        """Stop resource monitoring and finalize metrics."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        self._finalize_metrics()
        logger.info(f"[{self.task_name}] Resource monitoring stopped")
        
    def _monitor_loop(self):
        """Background loop to collect resource metrics."""
        while self._monitoring:
            try:
                # CPU memory (including children processes)
                cpu_mem_mb = self.process.memory_info().rss / 1024 / 1024
                self.cpu_memory_samples.append(cpu_mem_mb)
                
                # CPU percent
                cpu_percent = self.process.cpu_percent(interval=0.1)
                self.cpu_percent_samples.append(cpu_percent)
                
                # GPU memory (NVIDIA)
                if TORCH_AVAILABLE:
                    try:
                        gpu_mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
                        self.gpu_memory_samples.append(gpu_mem_mb)
                    except Exception as e:
                        logger.debug(f"GPU memory read failed: {e}")
                
                time.sleep(self.tracking_interval)
            except Exception as e:
                logger.warning(f"Error in resource monitoring: {e}")
    
    def _finalize_metrics(self):
        """Calculate final metrics from collected samples."""
        end_time = datetime.now()
        self.metrics['end_time'] = end_time.isoformat()
        self.metrics['duration_seconds'] = (end_time - datetime.fromisoformat(self.metrics['start_time'])).total_seconds()
        
        # CPU memory stats
        if self.cpu_memory_samples:
            self.metrics['cpu_memory_peak_mb'] = max(self.cpu_memory_samples)
            self.metrics['cpu_memory_avg_mb'] = sum(self.cpu_memory_samples) / len(self.cpu_memory_samples)
        
        # GPU memory stats
        if self.gpu_memory_samples:
            self.metrics['gpu_memory_peak_mb'] = max(self.gpu_memory_samples)
            self.metrics['gpu_memory_avg_mb'] = sum(self.gpu_memory_samples) / len(self.gpu_memory_samples)
        
        # CPU percent stats
        if self.cpu_percent_samples:
            self.metrics['cpu_percent_avg'] = sum(self.cpu_percent_samples) / len(self.cpu_percent_samples)
        
        self.metrics['status'] = 'completed'
    
    def log_metrics(self, status: str = 'completed'):
        """Log collected metrics to file and logger."""
        self.metrics['status'] = status
        
        # Format for readability
        log_msg = (
            f"\n{'='*70}\n"
            f"RESOURCE METRICS: {self.task_name} (video_id={self.video_id})\n"
            f"{'='*70}\n"
            f"Status: {status}\n"
            f"Duration: {self.metrics['duration_seconds']:.2f}s\n"
            f"\nCPU Memory:\n"
            f"  Peak: {self.metrics['cpu_memory_peak_mb']:.2f} MB\n"
            f"  Average: {self.metrics['cpu_memory_avg_mb']:.2f} MB\n"
            f"  CPU Utilization: {self.metrics['cpu_percent_avg']:.1f}%\n"
            f"\nGPU Memory (NVIDIA):\n"
            f"  Peak: {self.metrics['gpu_memory_peak_mb']:.2f} MB\n"
            f"  Average: {self.metrics['gpu_memory_avg_mb']:.2f} MB\n"
            f"{'='*70}\n"
        )
        
        logger.info(log_msg)
        self._save_metrics_to_file()
        
    def _save_metrics_to_file(self):
        """Save metrics to CSV file for cost analysis."""
        metrics_file = os.path.join(
            os.path.dirname(__file__),
            '..', 'logs',
            'resource_metrics.csv'
        )
        
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        
        # Check if file exists to determine if we need header
        file_exists = os.path.exists(metrics_file)
        
        try:
            import csv
            with open(metrics_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'task_name', 'video_id', 'duration_seconds',
                    'cpu_memory_peak_mb', 'cpu_memory_avg_mb', 'cpu_percent_avg',
                    'gpu_memory_peak_mb', 'gpu_memory_avg_mb', 'status'
                ])
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({
                    'timestamp': self.metrics['start_time'],
                    'task_name': self.metrics['task_name'],
                    'video_id': self.metrics['video_id'],
                    'duration_seconds': f"{self.metrics['duration_seconds']:.2f}",
                    'cpu_memory_peak_mb': f"{self.metrics['cpu_memory_peak_mb']:.2f}",
                    'cpu_memory_avg_mb': f"{self.metrics['cpu_memory_avg_mb']:.2f}",
                    'cpu_percent_avg': f"{self.metrics['cpu_percent_avg']:.1f}",
                    'gpu_memory_peak_mb': f"{self.metrics['gpu_memory_peak_mb']:.2f}",
                    'gpu_memory_avg_mb': f"{self.metrics['gpu_memory_avg_mb']:.2f}",
                    'status': self.metrics['status'],
                })
            logger.debug(f"Metrics saved to {metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save metrics to CSV: {e}")
    
    def get_metrics(self) -> Dict:
        """Return collected metrics dictionary."""
        return self.metrics.copy()


class ResourceMonitorContext:
    """Context manager for automatic resource monitoring."""
    
    def __init__(self, task_name: str, video_id: int = None, tracking_interval: float = 2.0):
        from .config import RESOURCE_MONITOR
        self._enabled = RESOURCE_MONITOR
        self.monitor = ResourceMonitor(task_name, video_id, tracking_interval) if self._enabled else None
    
    def __enter__(self):
        if self._enabled:
            self.monitor.start()
        return self.monitor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._enabled:
            self.monitor.stop()
            status = 'failed' if exc_type else 'completed'
            self.monitor.log_metrics(status=status)
        if exc_type:
            logger.error(f"Task failed with exception: {exc_val}", exc_info=(exc_type, exc_val, exc_tb))
