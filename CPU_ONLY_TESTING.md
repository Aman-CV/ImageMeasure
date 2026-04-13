# CPU-Only Mode for Testing

Force all AI models to use CPU instead of GPU for testing purposes.

## How to Enable CPU-Only Mode

### Option 1: Set Environment Variable (Recommended)

**On Windows (PowerShell):**
```powershell
$env:IMAGEMEASURE_CPU_ONLY = "1"
python manage.py process_tasks
```

**On Windows (Command Prompt):**
```cmd
set IMAGEMEASURE_CPU_ONLY=1
python manage.py process_tasks
```

**On Linux/Mac:**
```bash
export IMAGEMEASURE_CPU_ONLY=1
python manage.py process_tasks
```

### Option 2: Set in Django Settings

Add to `Measure/Measure/settings.py`:
```python
import os
os.environ['IMAGEMEASURE_CPU_ONLY'] = '1'
```

### Option 3: One-Time Command

**Windows (PowerShell):**
```powershell
(C:\Users\AmanGautam\anaconda3\shell\condabin\conda-hook.ps1) ; (conda activate py38)
$env:IMAGEMEASURE_CPU_ONLY = "1"
cd Measure
python manage.py process_tasks
```

## What Gets Forced to CPU

When `IMAGEMEASURE_CPU_ONLY=1`, the following models use CPU:

1. **YOLO Detection Models** (4 locations):
   - `task.py` - YOLOv8m-pose for broad_jump
   - `sit_and_throw_helper.py` - YOLOv8x for ball detection
   - `plank_helper.py` - YOLOv8m-pose for plank pose
   - `checkpoint_crossing.py` - YOLOv8m, YOLOv8m-pose, YOLOv8x for timing

2. **SAM Segmentation Model**:
   - `helper.py` - Segment Anything Model (SAM vit_b)

## Verify CPU-Only is Active

### Check in Logs
```bash
tail -f logs/django.log | grep -i "cpu"
```

### Python Script
```python
import os
os.environ['IMAGEMEASURE_CPU_ONLY'] = '1'

from homography_app.config import CPU_ONLY, get_device, get_torch_device
print(f"CPU_ONLY Mode: {CPU_ONLY}")
print(f"Device for YOLO: {get_device()}")
print(f"Device for PyTorch: {get_torch_device()}")
```

Expected output:
```
CPU_ONLY Mode: True
Device for YOLO: cpu
Device for PyTorch: cpu(cpu)
```

## Disable CPU-Only Mode

**Windows (PowerShell):**
```powershell
$env:IMAGEMEASURE_CPU_ONLY = "0"
# or remove it
Remove-Item Env:IMAGEMEASURE_CPU_ONLY
```

**Linux/Mac:**
```bash
unset IMAGEMEASURE_CPU_ONLY
# or
export IMAGEMEASURE_CPU_ONLY=0
```

## Files Modified

- `homography_app/config.py` - NEW centralized device configuration
- `homography_app/task.py` - YOLO forced to CPU
- `homography_app/sit_and_throw_helper.py` - YOLO forced to CPU
- `homography_app/plank_helper.py` - YOLO forced to CPU
- `homography_app/checkpoint_crossing.py` - YOLO forced to CPU
- `homography_app/helper.py` - SAM forced to CPU

## Performance Notes

**CPU Mode Advantages:**
- ✅ Test without GPU hardware
- ✅ Debugging easier (single device)
- ✅ No GPU memory conflicts
- ✅ Portable (works on any machine)

**CPU Mode Disadvantages:**
- ❌ Much slower processing (~5-10x slower)
- ❌ Higher CPU & memory usage
- ❌ Not suitable for production

## Testing Workflow

```bash
# 1. Enable CPU-only mode
$env:IMAGEMEASURE_CPU_ONLY = "1"

# 2. Start your Django server
cd Measure
python manage.py runserver 0.0.0.0:8000

# 3. Upload test videos and verify logic works
# (Don't worry about speed - you're just testing logic)

# 4. When done, disable and use GPU for production
$env:IMAGEMEASURE_CPU_ONLY = "0"
```

## Configuration File

See `homography_app/config.py` for the implementation:

```python
CPU_ONLY = os.getenv('IMAGEMEASURE_CPU_ONLY', '0') == '1'

def get_device():
    """Returns 'cpu' if CPU_ONLY=True, else 'gpu' or 'cuda'"""
    if CPU_ONLY:
        return 'cpu'
    # ... rest of logic

def get_torch_device():
    """Returns torch.device('cpu') if CPU_ONLY=True"""
    import torch
    if CPU_ONLY:
        return torch.device('cpu')
    # ... rest of logic
```

## FAQ

**Q: Why is processing so slow?**  
A: CPU processing is 5-10x slower than GPU. Use `CPU_ONLY=1` only for testing.

**Q: Do I need to change code?**  
A: No! Just set the environment variable. All changes are backward compatible.

**Q: Can I still use GPU?**  
A: Yes! Just unset the `IMAGEMEASURE_CPU_ONLY` variable.

**Q: What if CPU-ONLY doesn't work?**  
A: Check that you set the environment variable BEFORE importing the models. The config is read at import time.

**Q: Can I debug step-by-step with CPU mode?**  
A: Yes! CPU mode makes debugging easier because no GPU switching occurs.
