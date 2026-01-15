import subprocess
import time
import logging
from datetime import datetime
INTERVAL_SECONDS = 60
NUM_GPUS = 4

logging.basicConfig(
    filename='gpu_memory.log', 
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s|%(name)s|%(message)s")
logger = logging.getLogger("gpumemory")

def get_gpu_memory():
    """Query GPU memory usage using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,memory.free,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split("\n")
    except FileNotFoundError:
        return None, "nvidia-smi not found. Is NVIDIA driver installed?"
    except subprocess.CalledProcessError as e:
        return None, f"Error running nvidia-smi: {e}"

def log_gpu_memory(logger):
    """Log current GPU memory stats."""
    gpu_data = get_gpu_memory()

    if gpu_data is None:
        logger.error("Failed to query GPU data")
        return False

    if isinstance(gpu_data, tuple):
        logger.error(gpu_data[1])
        return False

    for line in gpu_data[:NUM_GPUS]:
        values = [v.strip() for v in line.split(",")]
        gpu_index = values[0]
        used_mb = values[1]
        total_mb = values[2]
        free_mb = values[3]
        util_pct = values[4]

        logger.info(
            f"GPU {gpu_index}: "
            f"Used={used_mb}MB, "
            f"Total={total_mb}MB, "
            f"Free={free_mb}MB, "
            f"Utilization={util_pct}%"
        )

    return True

def main():

    try:
        while True:
            log_gpu_memory(logger)
            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")

if __name__ == "__main__":
    main()

