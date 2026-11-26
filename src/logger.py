import os
from datetime import datetime
from src.paths import LOG_FILE


def log(log_message, begin=False):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode="a", encoding="utf-8") as log_file:
        if (begin):
            log_file.write("-" * 50)

        log_file.write(f"[{timestamp}] {log_message}\n")

        log_file.flush()
        os.fsync(log_file.fileno())