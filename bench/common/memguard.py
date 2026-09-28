"""Keep the benchmark inside the machine's memory: new work waits while available memory
is low, and a background thread logs usage."""
import os
import threading
import time

import psutil

MIN_AVAILABLE_GB = float(os.environ.get("BENCH_MIN_AVAILABLE_GB", "3"))
PEAK = {"used_gb": 0.0, "min_available_gb": 1e9}


def available_gb():
    return psutil.virtual_memory().available / 2**30


def wait():
    """Block until at least MIN_AVAILABLE_GB is available."""
    while available_gb() < MIN_AVAILABLE_GB:
        time.sleep(2)


def start_monitor(path, interval=15):
    def loop():
        while True:
            memory = psutil.virtual_memory()
            PEAK["used_gb"] = max(PEAK["used_gb"], memory.used / 2**30)
            PEAK["min_available_gb"] = min(PEAK["min_available_gb"], memory.available / 2**30)
            with open(path, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} available={memory.available / 2**30:.1f}GB used={memory.used / 2**30:.1f}GB\n")
            time.sleep(interval)
    threading.Thread(target=loop, daemon=True).start()
