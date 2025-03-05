#logging_config.py
import logging
import os

def configure_logging():
    """
    Configure logging for the entire application to control verbosity
    while ensuring detection details are properly recorded.
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # File handler for complete logs
            logging.FileHandler('logs/wildeye_complete.log'),
            # Console handler with filtered output
            logging.StreamHandler()
        ]
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