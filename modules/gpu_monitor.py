import time
import pynvml
from config import MAX_GPU_TEMP, CHECK_INTERVAL

def get_gpu_temp():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        pynvml.nvmlShutdown()
        return temp
    except Exception:
        return None


def wait_for_cooldown():
    while True:
        gpu_t = get_gpu_temp()
        if not gpu_t or gpu_t <= MAX_GPU_TEMP:
            break
        print(f"[WAIT] GPU {gpu_t}°C > {MAX_GPU_TEMP}°C, waiting {CHECK_INTERVAL}s...")
        time.sleep(CHECK_INTERVAL)