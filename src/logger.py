import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"  # Generating the log file name based on current date and time
logs_path = os.path.join(
    os.getcwd(), "logs", LOG_FILE
)  # Creating the path for the log file
os.makedirs(logs_path, exist_ok=True)  # Creating the logs directory if it doesn't exist

LOG_FILE_PATH = os.path.join(
    logs_path, LOG_FILE  # Creating the full path for the log file
)

logging.basicConfig(
    filename=LOG_FILE_PATH,  # Setting the log file path
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s %(message)s",  # Configuring the log message format
    level=logging.INFO,  # Setting the logging level to INFO
)

if __name__ == "__main__":
    logging.info("Logging has started")
