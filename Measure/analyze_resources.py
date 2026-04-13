#!/usr/bin/env python3
"""
AWS Cost Analysis Tool for ImageMeasure Video Processing
Analyzes resource_metrics.csv to estimate AWS costs
"""

import csv
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime

class AWSCostAnalyzer:
    """Analyze resource usage and estimate AWS costs."""
    
    # AWS Pricing (as of 2024 - adjust based on your region)
    # US East (N. Virginia) pricing
    AWS_PRICES = {
        'ec2_cpu_hour': 0.0116,  # t3.medium or similar
        'ec2_gpu_hour': 0.35,    # GPU instance (p3.2xlarge NVIDIA V100)
        'memory_gb_hour': 0.0001,  # Included in EC2, but tracked separately
    }
    
    def __init__(self, csv_file):
        """Initialize analyzer with CSV file path."""
        self.csv_file = csv_file
        self.metrics = []
        self.task_stats = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0,
            'avg_cpu_memory': 0,
            'avg_gpu_memory': 0,
            'avg_cpu_percent': 0,
            'peak_cpu_memory': 0,
            'peak_gpu_memory': 0,
        })
        
    def load_metrics(self):
        """Load metrics from CSV file."""
        if not os.path.exists(self.csv_file):
            print(f"Metrics file not found: {self.csv_file}")
            return False
            
        try:
            with open(self.csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.metrics.append(row)
            print(f"Loaded {len(self.metrics)} metric records")
            return True
        except Exception as e:
            print(f"Error loading metrics: {e}")
            return False
    
    def analyze(self):
        """Analyze collected metrics."""
        if not self.metrics:
            print("No metrics to analyze")
            return
        
        # Group by task type
        for metric in self.metrics:
            task_name = metric.get('task_name', 'unknown')
            try:
                duration = float(metric.get('duration_seconds', 0))
                cpu_mem = float(metric.get('cpu_memory_avg_mb', 0))
                gpu_mem = float(metric.get('gpu_memory_avg_mb', 0))
                cpu_percent = float(metric.get('cpu_percent_avg', 0))
                peak_cpu = float(metric.get('cpu_memory_peak_mb', 0))
                peak_gpu = float(metric.get('gpu_memory_peak_mb', 0))
                
                stats = self.task_stats[task_name]
                stats['count'] += 1
                stats['total_duration'] += duration
                stats['avg_cpu_memory'] = (stats['avg_cpu_memory'] * (stats['count'] - 1) + cpu_mem) / stats['count']
                stats['avg_gpu_memory'] = (stats['avg_gpu_memory'] * (stats['count'] - 1) + gpu_mem) / stats['count']
                stats['avg_cpu_percent'] = (stats['avg_cpu_percent'] * (stats['count'] - 1) + cpu_percent) / stats['count']
                stats['peak_cpu_memory'] = max(stats['peak_cpu_memory'], peak_cpu)
                stats['peak_gpu_memory'] = max(stats['peak_gpu_memory'], peak_gpu)
            except ValueError:
                continue
    
    def print_report(self):
        """Print analysis report."""
        print("\n" + "="*80)
        print("AWS RESOURCE ANALYSIS REPORT")
        print("="*80 + "\n")
        
        total_duration_hours = 0
        total_gpu_usage_hours = 0
        
        for task_name, stats in sorted(self.task_stats.items()):
            print(f"\nTask: {task_name}")
            print(f"  Executions: {stats['count']}")
            print(f"  Total Duration: {stats['total_duration']:.2f} seconds ({stats['total_duration']/3600:.4f} hours)")
            print(f"  \nMemory Usage:")
            print(f"    CPU Average: {stats['avg_cpu_memory']:.2f} MB")
            print(f"    CPU Peak: {stats['peak_cpu_memory']:.2f} MB")
            print(f"    GPU Average: {stats['avg_gpu_memory']:.2f} MB")
            print(f"    GPU Peak: {stats['peak_gpu_memory']:.2f} MB")
            print(f"  CPU Utilization: {stats['avg_cpu_percent']:.1f}%")
            
            task_gpu_hours = (stats['total_duration'] / 3600) if stats['avg_gpu_memory'] > 0 else 0
            total_duration_hours += stats['total_duration'] / 3600
            total_gpu_usage_hours += task_gpu_hours
        
        print("\n" + "-"*80)
        print("COST ESTIMATION")
        print("-"*80)
        
        # Calculate costs
        cpu_cost = total_duration_hours * self.AWS_PRICES['ec2_cpu_hour']
        gpu_cost = total_gpu_usage_hours * self.AWS_PRICES['ec2_gpu_hour']
        total_cost = cpu_cost + gpu_cost
        
        print(f"\nCPU Instance Hours: {total_duration_hours:.4f} hours")
        print(f"  Estimated Cost: ${cpu_cost:.2f}")
        
        if total_gpu_usage_hours > 0:
            print(f"\nGPU Instance Hours: {total_gpu_usage_hours:.4f} hours")
            print(f"  Estimated Cost: ${gpu_cost:.2f}")
        
        print(f"\nTotal Estimated Cost: ${total_cost:.2f}")
        
        if self.metrics:
            cost_per_video = total_cost / len(self.metrics)
            print(f"Cost per Video: ${cost_per_video:.4f}")
        
        print("\n" + "="*80)
    
    def export_summary(self, output_file=None):
        """Export analysis summary to file."""
        if not output_file:
            output_file = os.path.join(
                os.path.dirname(self.csv_file),
                'resource_summary.txt'
            )
        
        try:
            import sys
            from io import StringIO
            
            # Capture print output
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            self.print_report()
            
            sys.stdout = old_stdout
            report = captured_output.getvalue()
            
            with open(output_file, 'w') as f:
                f.write(report)
                f.write(f"\n\nGenerated: {datetime.now().isoformat()}\n")
            
            print(f"Summary saved to: {output_file}")
        except Exception as e:
            print(f"Error exporting summary: {e}")


if __name__ == "__main__":
    import sys
    
    # Find metrics file
    metrics_file = Path(__file__).parent / 'logs' / 'resource_metrics.csv'
    
    if len(sys.argv) > 1:
        metrics_file = Path(sys.argv[1])
    
    print(f"Analyzing metrics from: {metrics_file}")
    
    analyzer = AWSCostAnalyzer(str(metrics_file))
    if analyzer.load_metrics():
        analyzer.analyze()
        analyzer.print_report()
        analyzer.export_summary()
