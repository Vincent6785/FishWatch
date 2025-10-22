import time
import pynvml
from config import MAX_GPU_TEMP, CHECK_INTERVAL

def get_gpu_temp() -> float | None:
    """
    Récupère la température actuelle du GPU à l'aide de NVML.

    Returns:
        Température du GPU en degrés Celsius, ou None si la lecture échoue.
    """
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        pynvml.nvmlShutdown()
        return temp
    except pynvml.NVMLError as e:
        print(f"[WARN] Unable to read GPU temperature: {e}")
        return None
    except Exception:
        return None


def wait_for_cooldown() -> None:
    """
    Attend que la température du GPU descende en dessous du seuil MAX_GPU_TEMP.
    Effectue des pauses régulières définies par CHECK_INTERVAL.
    """
    while True:
        gpu_temp = get_gpu_temp()
        if gpu_temp is None:
            # Si la température ne peut pas être lue, on suppose qu'on peut continuer
            print("[WARN] Could not retrieve GPU temperature, skipping cooldown check.")
            break

        if gpu_temp <= MAX_GPU_TEMP:
            break

        print(f"[WAIT] GPU temperature {gpu_temp}°C > {MAX_GPU_TEMP}°C, waiting {CHECK_INTERVAL}s...")
        time.sleep(CHECK_INTERVAL)
