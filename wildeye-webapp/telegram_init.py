# telegram_init.py
import os
import logging
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

def setup_telegram_environment():
    """
    Set up Telegram environment variables
    """
    # Load environment variables from .env file if present
    load_dotenv()
    
    # Check for required Telegram environment variables
    telegram_bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    
    if not telegram_bot_token:
        logger.warning("TELEGRAM_BOT_TOKEN not set in environment variables")
        logger.warning("Telegram notifications will not work without a valid bot token")
    else:
        logger.info("Telegram bot token found in environment variables")
    
    # Check for emergency contact
    emergency_contact = os.environ.get('EMERGENCY_CONTACT')
    if not emergency_contact:
        default_contact = "Forest Dept: +XX-XXXX-XXXXXX"
        os.environ['EMERGENCY_CONTACT'] = default_contact
        logger.info(f"EMERGENCY_CONTACT not set, using default: {default_contact}")
    
    # Return status
    return {
        'has_token': bool(os.environ.get('TELEGRAM_BOT_TOKEN')),
        'emergency_contact': os.environ.get('EMERGENCY_CONTACT')
    }