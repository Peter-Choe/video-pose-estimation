import logging
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("pose_inference")
logger.setLevel(logging.INFO)

# Remove existing FileHandlers to avoid duplicates
for h in logger.handlers[:]:
    if isinstance(h, logging.FileHandler):
        logger.removeHandler(h)

# Formatter
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')

# General log file
general_log_file = os.path.abspath(f"{log_dir}/video_pose_inference.log")
file_handler = logging.FileHandler(general_log_file)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console output
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Add custom filter class
class MetricsOnlyFilter(logging.Filter):
    def filter(self, record):
        return getattr(record, "metrics_only", False)

# Metrics log file (filtered)
metrics_log_file = os.path.abspath(f"{log_dir}/video_inference_times.log")
metrics_handler = logging.FileHandler(metrics_log_file)
metrics_handler.setFormatter(logging.Formatter('%(message)s'))  # raw format
metrics_handler.setLevel(logging.INFO)
metrics_handler.addFilter(MetricsOnlyFilter())  #  Attach filter 
logger.addHandler(metrics_handler)


