# Resource Monitoring & AWS Cost Analysis Guide

## Overview
This system automatically tracks CPU and GPU (NVIDIA CUDA) resource usage during video processing tasks. All metrics are logged to a CSV file for cost estimation and optimization analysis.

## How It Works

### 1. **Automatic Monitoring**
Each video processing task is wrapped with `ResourceMonitorContext`, which:
- Monitors CPU memory usage every 2 seconds
- Tracks GPU memory allocation in real-time
- Records CPU utilization percentage
- Captures peak and average values
- Automatically logs metrics on task completion

### 2. **Supported Tasks**
The following video processing tasks are monitored:
- `broad_jump` - Jump distance measurement
- `sit_and_throw` - Throw distance with ball detection
- `sit_and_reach` - Flexibility measurement
- `plank` - Plank endurance tracking
- `6x15_dash` - Sprint timing

### 3. **Metrics Collected**

For each task, the system records:

| Metric | Unit | Description |
|--------|------|-------------|
| `timestamp` | ISO 8601 | When the task started |
| `task_name` | string | Type of processing task |
| `video_id` | int | Database ID of the video |
| `duration_seconds` | seconds | Total processing time |
| `cpu_memory_peak_mb` | MB | Maximum RAM used |
| `cpu_memory_avg_mb` | MB | Average RAM usage |
| `cpu_percent_avg` | % | Average CPU utilization |
| `gpu_memory_peak_mb` | MB | Maximum VRAM used |
| `gpu_memory_avg_mb` | MB | Average VRAM usage |
| `status` | string | 'completed' or 'failed' |

## Output Files

### Resource Metrics CSV
- **Location**: `Measure/logs/resource_metrics.csv`
- **Format**: Comma-separated values with headers
- **Updated**: After each completed task
- **Use**: Analysis and cost estimation

### Example CSV Content
```
timestamp,task_name,video_id,duration_seconds,cpu_memory_peak_mb,cpu_memory_avg_mb,cpu_percent_avg,gpu_memory_peak_mb,gpu_memory_avg_mb,status
2024-04-13T10:15:32.123456,broad_jump,1234,45.23,1024.50,856.30,78.5,2048.75,1856.40,completed
2024-04-13T10:16:18.456789,sit_and_throw,1235,32.10,892.30,720.15,72.3,1536.20,1280.50,completed
```

## Analysis & Cost Estimation

### Running the Analysis Script

```bash
cd Measure
python analyze_resources.py
```

Or with a custom metrics file:
```bash
python analyze_resources.py /path/to/resource_metrics.csv
```

### What the Analysis Shows

The analysis script (`analyze_resources.py`) generates a report with:

1. **Per-Task Statistics**
   - Number of executions
   - Total duration
   - Average/peak memory usage
   - CPU utilization

2. **Cost Estimation**
   - CPU instance hours
   - GPU instance hours
   - Estimated AWS costs
   - Cost per video processed

### Example Report Output

```
================================================================================
AWS RESOURCE ANALYSIS REPORT
================================================================================

Task: broad_jump
  Executions: 150
  Total Duration: 6750.00 seconds (1.8750 hours)
  
Memory Usage:
    CPU Average: 856.30 MB
    CPU Peak: 1024.50 MB
    GPU Average: 1856.40 MB
    GPU Peak: 2048.75 MB
  CPU Utilization: 78.5%

Task: sit_and_throw
  Executions: 120
  Total Duration: 3852.00 seconds (1.0700 hours)
  ...

--------------------------------------------------------------------------------
COST ESTIMATION
--------------------------------------------------------------------------------

CPU Instance Hours: 57.2500 hours
  Estimated Cost: $0.66

GPU Instance Hours: 57.2500 hours
  Estimated Cost: $20.04

Total Estimated Cost: $20.70
Cost per Video: $0.09

================================================================================
```

## AWS Pricing Reference

Default pricing (adjust based on your region and instance type):

| Resource | Pricing |
|----------|---------|
| EC2 t3.medium (CPU) | $0.0116/hour |
| GPU Instance (p3.2xlarge) | $0.35/hour |
| Storage (S3) | $0.023/GB/month |
| Data Transfer | $0.02/GB (outbound) |

**Note**: Modify prices in `analyze_resources.py` if using different AWS regions or instance types.

## Integration with Logging

### View Real-Time Logs

Logs are written to Django's logger with detailed resource information:

```bash
# View logs in real-time
tail -f logs/django.log

# Search for specific task
grep "broad_jump" logs/django.log

# View resource logs for a video
grep "video=1234" logs/django.log
```

### Log Format
```
[2024-04-13 10:15:32,123] INFO [homography_app] broad_jump started: video=1234 test=test_id assessment=assessment_id

======================================================================
RESOURCE METRICS: broad_jump (video_id=1234)
======================================================================
Status: completed
Duration: 45.23s

CPU Memory:
  Peak: 1024.50 MB
  Average: 856.30 MB
  CPU Utilization: 78.5%

GPU Memory (NVIDIA):
  Peak: 2048.75 MB
  Average: 1856.40 MB
======================================================================
```

## Requirements

### Python Packages
```
psutil>=5.8.0  # System resource monitoring
torch>=1.9.0   # GPU support (if using NVIDIA)
```

### Install
```bash
pip install psutil torch
```

### Verify GPU Support
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

## Customization

### Adjust Monitoring Interval

To change how frequently resources are sampled (default: 2 seconds):

```python
# In your task function
monitor = ResourceMonitorContext('task_name', video_id, tracking_interval=1.0)  # 1 second interval
```

### Add Custom Metrics

To track additional metrics, modify `resource_monitor.py`:

```python
def _monitor_loop(self):
    while self._monitoring:
        # ... existing code ...
        
        # Add custom metric
        your_metric = get_custom_metric()
        self.your_metric_samples.append(your_metric)
```

### Change AWS Pricing

Edit `analyze_resources.py`:

```python
AWS_PRICES = {
    'ec2_cpu_hour': 0.0116,      # Change this
    'ec2_gpu_hour': 0.35,        # Change this  
    'memory_gb_hour': 0.0001,
}
```

## Troubleshooting

### GPU Memory Not Tracking

**Problem**: GPU memory shows 0 MB

**Solution**:
1. Verify PyTorch is installed: `pip install torch torchvision`
2. Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
3. Ensure GPU is being used in code

### CSV File Permission Error

**Problem**: "Permission denied" when writing to CSV

**Solution**:
1. Check folder permissions: `ls -la logs/`
2. Create logs folder if missing: `mkdir -p logs/`
3. Ensure write permissions: `chmod 755 logs/`

### Missing Metrics Data

**Problem**: Some rows have 0 or missing values

**Solution**:
1. Check if task failed (status='failed')
2. Verify monitoring started: Look for "Resource monitoring started" in logs
3. Check if task duration was too short (< 1 second)

## Best Practices

### 1. **Regular Analysis**
Run analysis weekly or after processing batches:
```bash
python analyze_resources.py > cost_report_$(date +%Y%m%d).txt
```

### 2. **Archive Old Data**
Keep only recent metrics for faster analysis:
```bash
# Keep only last 30 days
tail -n +1 resource_metrics.csv | head -1 > temp.csv
tail -n +2 resource_metrics.csv | awk -v cutoff=$(date -d "30 days ago" +%s) 'BEGIN{FS=","} {print $1}' | head -20 >> temp.csv
```

### 3. **Compare Task Performance**
Use the CSV to identify inefficient tasks:
```python
import pandas as pd
df = pd.read_csv('logs/resource_metrics.csv')
print(df.groupby('task_name')[['duration_seconds', 'gpu_memory_peak_mb']].mean())
```

### 4. **Set up Alerts**
Monitor for resource spikes:
```python
df = pd.read_csv('logs/resource_metrics.csv')
spike = df[df['gpu_memory_peak_mb'] > 3000]  # Alert if > 3GB
print("GPU Spike Alert:", spike)
```

## Example Workflow

### Day 1: Run Processing
```bash
# Your normal video processing runs
# Metrics are automatically collected to logs/resource_metrics.csv
```

### Day 2: Analyze Results
```bash
python analyze_resources.py
# Review the cost report
```

### Day 3: Optimize
```bash
# Identify expensive tasks
# Adjust encoding settings or batch sizes
# Re-run analysis to measure improvement
```

## Advanced: Database Storage

For long-term analysis, consider storing metrics in your Django database:

```python
# Add to your models.py
class ResourceMetrics(models.Model):
    video = models.ForeignKey(PetVideos, on_delete=models.CASCADE)
    task_name = models.CharField(max_length=50)
    duration_seconds = models.FloatField()
    cpu_memory_peak_mb = models.FloatField()
    gpu_memory_peak_mb = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
```

Then modify `resource_monitor.py` to save to database as well as CSV.

## Support & Questions

For issues or questions about resource monitoring:
1. Check Django logs: `Measure/logs/django.log`
2. Verify GPU setup: `nvidia-smi`
3. Review this documentation
4. Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`

---
**Last Updated**: April 2024  
**Version**: 1.0  
**Status**: Production Ready
