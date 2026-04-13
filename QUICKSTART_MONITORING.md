# Quick Start: Resource Monitoring for AWS Cost Analysis

## Installation (1 minute)

### 1. Install Required Packages
```bash
cd Measure
pip install psutil
# If not already installed for GPU support:
pip install torch torchvision
```

### 2. Verify Installation
```bash
python -c "import psutil; print('psutil OK')"
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
```

## Automatic Usage (No changes needed!)

Your video processing tasks now automatically track and log resources. No code changes required!

### The monitoring is active when you run:
```bash
python manage.py process_tasks
```

Or when tasks are triggered through your Django views.

## View Statistics (5 minutes)

### Check Logs in Real-Time
```bash
# Watch logs as tasks run
tail -f logs/django.log | grep "RESOURCE METRICS"
```

### Generate Cost Report
```bash
cd Measure
python analyze_resources.py
```

## Output Files

After running your first task, you'll have:

```
Measure/
├── logs/
│   └── resource_metrics.csv          ← Raw metrics (opens in Excel)
│   └── resource_summary.txt          ← Cost report
│   └── django.log                    ← Detailed logs
└── analyze_resources.py
```

## What You'll See

### In Logs (During Processing)
```
[INFO] broad_jump started: video=1234 ...

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

### In CSV File (for analysis)
```
timestamp,task_name,video_id,duration_seconds,cpu_memory_peak_mb,...
2024-04-13T10:15:32,broad_jump,1234,45.23,1024.50,856.30,...
```

### In Analysis Report
```
AWS RESOURCE ANALYSIS REPORT

Task: broad_jump
  Executions: 150
  Total Duration: 6750.00 seconds (1.8750 hours)
  CPU Average: 856.30 MB
  GPU Peak: 2048.75 MB

COST ESTIMATION
CPU Instance Hours: 57.25 hours
  Estimated Cost: $0.66

GPU Instance Hours: 57.25 hours
  Estimated Cost: $20.04

Total Estimated Cost: $20.70
Cost per Video: $0.09
```

## Common Tasks

### Generate Weekly Report
```bash
# Save report with date
python analyze_resources.py > logs/cost_report_$(date +%Y%m%d).txt
cat logs/cost_report_*.txt
```

### View Latest 10 Processing Tasks
```bash
tail -10 logs/resource_metrics.csv
```

### Find Most Resource-Intensive Tasks
```bash
# On Windows
python -c "
import pandas as pd
df = pd.read_csv('logs/resource_metrics.csv')
expensive = df.nlargest(5, 'gpu_memory_peak_mb')
print(expensive[['task_name', 'video_id', 'gpu_memory_peak_mb', 'duration_seconds']])
"
```

### Track Single Video
```bash
grep ",1234," logs/resource_metrics.csv  # Replace 1234 with video_id
```

## Customization

### Change GPU Tracking Interval
Edit: `homography_app/resource_monitor.py`
```python
# Change from 2.0 to 1.0 for more frequent sampling
ResourceMonitorContext('task_name', video_id, tracking_interval=1.0)
```

### Update AWS Pricing
Edit: `analyze_resources.py`
```python
AWS_PRICES = {
    'ec2_cpu_hour': 0.0116,  # Update me
    'ec2_gpu_hour': 0.35,    # Update me
}
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| GPU shows 0 MB | Run: `python -c "import torch; print(torch.cuda.is_available())"` |
| No logs file | Create: `mkdir -p logs/` |
| Permission denied | Run: `chmod 755 logs/` |
| Missing psutil | Run: `pip install psutil` |

## Files Modified

✅ `homography_app/task.py` - Added resource monitoring to all tasks
✅ `homography_app/resource_monitor.py` - NEW monitoring module
✅ `analyze_resources.py` - NEW analysis script
✅ `RESOURCE_MONITORING.md` - Detailed documentation

## Next Steps

1. **Run a video processing task** - Metrics collect automatically
2. **Wait for completion** - Check logs for resource metrics
3. **Generate report** - `python analyze_resources.py`
4. **Review costs** - Adjust settings if needed

## Questions?

See `RESOURCE_MONITORING.md` for detailed documentation and advanced usage.

---
**Created**: April 2024  
**Easy to use**: Yes, automatic!  
**Cost estimation**: Ready to use!
