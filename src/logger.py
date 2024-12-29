import logging
import os
from datetime import datetime

# Generate a timestamped log file name in the format "MM_DD_YYYY_HH_MM_SS.log"
current_time = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
log_file_name = f"{current_time}.log"

# Define the directory for storing logs (in the "logs" folder inside the current working directory)
log_directory = os.path.join(os.getcwd(), "logs")

# Ensure the log directory exists; create it if it doesn't
os.makedirs(log_directory, exist_ok=True)

# Full path for the log file
log_file_path = os.path.join(log_directory, log_file_name)

# Configure the logging settings
logging.basicConfig(
    filename=log_file_path,  # Log file path
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Log message format
    level=logging.INFO,  # Logging level (INFO captures standard application flow and events)
)