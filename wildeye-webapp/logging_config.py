import logging
import os
from datetime import datetime

class TimeFormatter(logging.Formatter):
    """Custom formatter to use 12-hour time format instead of 24-hour"""
    
    def formatTime(self, record, datefmt=None):
        """Override to format timestamps in 12-hour format"""
        created_time = datetime.fromtimestamp(record.created)
        if datefmt:
            return created_time.strftime(datefmt)
        else:
            # Default to 12-hour format with AM/PM
            return created_time.strftime("%Y-%m-%d %I:%M:%S %p")

def configure_logging():
    """
    Configure logging for the entire application to control verbosity
    while ensuring detection details are properly recorded.
    Also ensures timestamps are in 12-hour format.
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create custom formatter with 12-hour time format
    formatter = TimeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler for complete logs
    file_handler = logging.FileHandler('logs/wildeye_complete.log')
    file_handler.setFormatter(formatter)
    
    # Console handler with filtered output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    
    # Configure specific loggers
    
    # Keep detection_handler at INFO level to ensure detection details are logged
    # This ensures the data is available for detection_history.html
    detection_handler_logger = logging.getLogger('detection_handler')
    detection_handler_logger.setLevel(logging.INFO)
    
    # Set YOLO related loggers to ERROR to hide detection model details
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    
    # Set werkzeug (Flask dev server) to WARNING to hide request logs
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    # Set other library loggers to higher levels to reduce noise
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)
    
    # Log the configuration
    logging.getLogger(__name__).info("Logging configured. Complete logs available in logs/wildeye_complete.log")
    logging.getLogger(__name__).info("Detection details logging enabled for detection_history.html")
    logging.getLogger(__name__).info("Timestamps now use 12-hour format (AM/PM)")