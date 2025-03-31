# sms_service.py
import logging
import os
import sys
from typing import Dict, Optional
from twilio.rest import Client
from datetime import datetime
from twilio.base.exceptions import TwilioRestException, TwilioException

# Configure logging
logger = logging.getLogger(__name__)

# Check if we need to create a custom formatter for this module
if not any(isinstance(handler.formatter, logging.Formatter) for handler in logger.handlers + logging.getLogger().handlers if hasattr(handler, 'formatter') and handler.formatter):
    # If no handlers or no formatters, set up a basic one
    # This ensures logs look good even if the module is used standalone
    class SMSTimeFormatter(logging.Formatter):
        """Custom formatter with 12-hour time format for SMS service logs"""
        def formatTime(self, record, datefmt=None):
            created_time = datetime.fromtimestamp(record.created)
            if datefmt:
                return created_time.strftime(datefmt)
            else:
                return created_time.strftime("%Y-%m-%d %I:%M:%S %p")
    
    formatter = SMSTimeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Load environment variables with defaults for development
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', '')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER', '')
EMERGENCY_CONTACT = os.environ.get('EMERGENCY_CONTACT', 'Forest Dept: XXX-XXX-XXXX')

# Optional configuration variables with defaults
SMS_MAX_LENGTH = int(os.environ.get('SMS_MAX_LENGTH', 1600))
SMS_RATE_LIMIT = int(os.environ.get('SMS_RATE_LIMIT', 100))  # Max SMS per day

# Log configuration status on module import
def _check_configuration():
    """Validate SMS configuration and log status"""
    missing_vars = []
    if not TWILIO_ACCOUNT_SID:
        missing_vars.append("TWILIO_ACCOUNT_SID")
    if not TWILIO_AUTH_TOKEN:
        missing_vars.append("TWILIO_AUTH_TOKEN")
    if not TWILIO_PHONE_NUMBER:
        missing_vars.append("TWILIO_PHONE_NUMBER")
    
    if missing_vars:
        logger.warning("Twilio is not fully configured. SMS notifications may not work.")
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        return False
    else:
        logger.info("Twilio configuration loaded successfully")
        logger.info(f"Using Twilio phone number: {TWILIO_PHONE_NUMBER}")
        logger.debug(f"SMS configured with max length: {SMS_MAX_LENGTH}, rate limit: {SMS_RATE_LIMIT}/day")
        return True

# Run configuration check
SMS_PROPERLY_CONFIGURED = _check_configuration()

# Animal-specific SMS templates
ANIMAL_SMS_TEMPLATES = {
    'default': """
WildEye Alert: Animal detected at {camera_name}.
Time: {timestamp}
Please exercise caution.
General precautions:
- Do not approach
- Keep a safe distance
- Contact authorities if needed
""",
    'tiger': """
DANGER: Tiger detected at {camera_name}.
Time: {timestamp}
IMMEDIATE ACTIONS:
- Stay indoors
- Secure doors/windows
- Keep pets inside
- Do NOT approach
- Contact forest dept: {emergency_contact}
""",
    'leopard': """
DANGER: Leopard detected at {camera_name}.
Time: {timestamp}
IMMEDIATE ACTIONS:
- Stay indoors
- Secure doors/windows
- Keep pets inside
- Do NOT approach
- Contact forest dept: {emergency_contact}
""",
    'elephant': """
WARNING: Elephant detected at {camera_name}.
Time: {timestamp}
PRECAUTIONS:
- Stay indoors
- Remain quiet
- Avoid sudden movements
- Contact forest dept: {emergency_contact}
""",
    'bear': """
DANGER: Bear detected at {camera_name}.
Time: {timestamp}
IMMEDIATE ACTIONS:
- Stay indoors
- Secure food sources
- Do NOT approach
- Contact forest dept: {emergency_contact}
""",
    'wild boar': """
CAUTION: Wild Boar detected at {camera_name}.
Time: {timestamp}
PRECAUTIONS:
- Keep pets/children away
- Stay indoors if nearby
- Contact authorities if aggressive: {emergency_contact}
""",
    'wild buffalo': """
DANGER: Wild Buffalo detected at {camera_name}.
Time: {timestamp}
IMMEDIATE ACTIONS:
- Stay indoors
- Keep far away
- Do NOT approach
- Contact forest dept: {emergency_contact}
""",
    'lion': """
EXTREME DANGER: Lion detected at {camera_name}.
Time: {timestamp}
IMMEDIATE ACTIONS:
- Stay indoors
- Secure all entrances
- Contact emergency services: {emergency_contact}
- Alert neighbors
"""
}

# Broadcast alert templates
BROADCAST_SMS_TEMPLATES = {
    'default': """
NEARBY ALERT: Animal detected {distance}km from your location.
Time: {timestamp}
PRECAUTIONS:
- Be aware of surroundings
- Keep safe distance if spotted
- Contact authorities if needed

This alert is from another WildEye user's camera.
""",
    'tiger': """
NEARBY DANGER: Tiger detected {distance}km from your location.
Time: {timestamp}
PRECAUTIONS:
- Be vigilant in the area
- Do NOT approach if sighted
- Contact forest dept if spotted: {emergency_contact}

This alert is from another WildEye user's camera.
""",
    'leopard': """
NEARBY DANGER: Leopard detected {distance}km from your location.
Time: {timestamp}
PRECAUTIONS:
- Be vigilant in the area
- Do NOT approach if sighted
- Contact forest dept if spotted: {emergency_contact}

This alert is from another WildEye user's camera.
""",
    'elephant': """
NEARBY WARNING: Elephant detected {distance}km from your location.
Time: {timestamp}
PRECAUTIONS:
- Be cautious in the area
- Keep significant distance if spotted
- Contact forest dept if sighted: {emergency_contact}

This alert is from another WildEye user's camera.
""",
    'bear': """
NEARBY DANGER: Bear detected {distance}km from your location.
Time: {timestamp}
PRECAUTIONS:
- Be vigilant in the area
- Keep food secure outdoors
- Contact forest dept if sighted: {emergency_contact}

This alert is from another WildEye user's camera.
""",
    'wild boar': """
NEARBY CAUTION: Wild Boar detected {distance}km from your location.
Time: {timestamp}
PRECAUTIONS:
- Be cautious in the area
- Keep pets supervised outdoors
- Contact authorities if seen: {emergency_contact}

This alert is from another WildEye user's camera.
""",
    'wild buffalo': """
NEARBY DANGER: Wild Buffalo detected {distance}km from your location.
Time: {timestamp}
PRECAUTIONS:
- Be extremely cautious in the area
- Do NOT approach if sighted
- Contact forest dept if spotted: {emergency_contact}

This alert is from another WildEye user's camera.
""",
    'lion': """
NEARBY EXTREME DANGER: Lion detected {distance}km from your location.
Time: {timestamp}
PRECAUTIONS:
- Avoid area if possible
- Travel in vehicle with closed windows
- Contact emergency services if sighted: {emergency_contact}

This alert is from another WildEye user's camera.
"""
}

def format_date_dmy(date_obj):
    """
    Format a datetime object to day-month-year format with 12-hour time
    
    Args:
        date_obj: datetime object to format
        
    Returns:
        formatted_date: String in format "DD-MM-YYYY HH:MM:SS AM/PM"
    """
    if not date_obj:
        return None
    return date_obj.strftime("%d-%m-%Y %I:%M:%S %p")

def get_animal_sms_template(animal_type: str, is_broadcast: bool = False, distance: str = None) -> str:
    """
    Get the appropriate SMS template for the specified animal type.
    
    Args:
        animal_type: The type of animal detected
        is_broadcast: Whether this is a broadcast alert from another user's camera
        distance: Distance from user's location (for broadcast alerts)
        
    Returns:
        str: SMS template text for the animal
    """
    animal_type = animal_type.lower()
    
    # Use broadcast templates for broadcast alerts
    if is_broadcast:
        templates = BROADCAST_SMS_TEMPLATES
        
        # Default distance if not provided
        distance = distance or "unknown"
        
        # Check for specific animal matches
        for animal in templates:
            if animal in animal_type:
                return templates[animal].format(distance=distance, emergency_contact=EMERGENCY_CONTACT, timestamp="{timestamp}")
        
        # Return default template if no match
        return templates['default'].format(distance=distance, emergency_contact=EMERGENCY_CONTACT, timestamp="{timestamp}")
    else:
        templates = ANIMAL_SMS_TEMPLATES
        
        # Check for specific animal matches
        for animal in templates:
            if animal in animal_type:
                return templates[animal].format(camera_name="{camera_name}", emergency_contact=EMERGENCY_CONTACT, timestamp="{timestamp}")
        
        # Return default template if no match
        return templates['default'].format(camera_name="{camera_name}", emergency_contact=EMERGENCY_CONTACT, timestamp="{timestamp}")

def format_phone_number(phone_number: str, country_code: str = None) -> str:
    """
    Format a phone number to ensure it has the proper country code prefix.
    
    Args:
        phone_number: The phone number to format
        country_code: The country code to use if not already present (e.g., '+91')
        
    Returns:
        str: Properly formatted phone number with country code
    """
    if not phone_number:
        return None
        
    # If number already has + prefix, return as is
    if phone_number.startswith('+'):
        return phone_number
        
    # Strip any leading zeros from the phone number
    phone_number = phone_number.lstrip('0')
    
    # If country code is provided, use it (highest priority)
    if country_code:
        # Make sure country code starts with +
        if not country_code.startswith('+'):
            country_code = '+' + country_code
            
        # Return number with country code
        return f"{country_code}{phone_number}"
    
    # If no country code provided, check for common country codes in the number itself
    if phone_number.startswith('1') and len(phone_number) >= 11:  # US/Canada with country code already in number
        return '+' + phone_number
    elif phone_number.startswith('91') and len(phone_number) >= 12:  # India with country code already in number
        return '+' + phone_number
    
    # Default to India country code if no other information
    logger.warning(f"No country code provided for {phone_number}. Defaulting to India (+91).")
    return f"+91{phone_number}"

# Update in sms_service.py - fixed send_sms function
# Update in sms_service.py - fixed send_sms function
def send_sms(recipient: str, message: str = None, detection_data: Dict = None, country_code: str = None) -> bool:
    """
    Send an SMS notification using Twilio.
    
    Args:
        recipient: Phone number to send SMS to
        message: Message text (optional)
        detection_data: Data about the animal detection (optional)
        country_code: Country code to use if not included in the recipient number
        
    Returns:
        success: True if SMS was sent successfully
    """
    try:
        # Validate requirements
        if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
            logger.error("Missing Twilio credentials - TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set in environment variables")
            return False
            
        if not TWILIO_PHONE_NUMBER:
            logger.error("Missing TWILIO_PHONE_NUMBER environment variable")
            return False
            
        if not recipient:
            logger.error("No recipient phone number provided")
            return False
            
        # Format phone number with country code
        formatted_recipient = format_phone_number(recipient, country_code)
                
        logger.info(f"Preparing to send SMS to {formatted_recipient}")
        
        # If message is provided directly, use it
        if message:
            sms_message = message
        # Otherwise, generate message from detection data
        elif detection_data:
            # Check if this is a broadcast alert
            is_broadcast = detection_data.get('is_broadcast', False)
            
            # Get distance for broadcast alerts
            distance = None
            if is_broadcast:
                # Use distance_km if available (numeric value)
                if 'distance_km' in detection_data:
                    distance = detection_data['distance_km']
                # Otherwise use distance (might be pre-formatted)
                elif 'distance' in detection_data:
                    distance = detection_data['distance']
                
                logger.info(f"Processing broadcast alert with distance: {distance}")
            
            # Get animal type - check both field names
            animal_type = detection_data.get('detection_label', '')
            if not animal_type:
                animal_type = detection_data.get('type', '')
            if not animal_type:
                animal_type = 'unknown'
            
            # Get camera name
            camera_name = detection_data.get('camera_name', 'Unknown location')
            
            # Format timestamp
            timestamp_str = ""
            timestamp = detection_data.get('timestamp', datetime.now())
            
            # Check for formatted_date first (new field)
            if 'formatted_date' in detection_data:
                timestamp_str = detection_data['formatted_date']
            elif 'formatted_timestamp' in detection_data:
                timestamp_str = detection_data['formatted_timestamp']
            elif isinstance(timestamp, datetime):
                timestamp_str = format_date_dmy(timestamp)
            else:
                timestamp_str = str(timestamp)
            
            # Get template based on whether it's a broadcast alert
            if is_broadcast:
                template = get_animal_sms_template(animal_type, is_broadcast=True, distance=distance)
                sms_message = template.format(timestamp=timestamp_str)
            else:
                template = get_animal_sms_template(animal_type)
                sms_message = template.format(camera_name=camera_name, timestamp=timestamp_str)
        else:
            logger.error("Neither message nor detection data provided")
            return False
        
        # Initialize Twilio client
        try:
            logger.info("Initializing Twilio client")
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            
            # Validate credentials before attempting to send
            # This will throw an exception if credentials are invalid
            logger.debug("Validating Twilio credentials")
            try:
                account_info = client.api.accounts(TWILIO_ACCOUNT_SID).fetch()
                logger.info(f"Successfully authenticated with Twilio account: {account_info.friendly_name}")
            except TwilioRestException as auth_error:
                if auth_error.code == 20003:  # Authentication error
                    logger.error("Twilio authentication failed - invalid Account SID or Auth Token")
                    return False
                raise
            
            # Limit message length for SMS
            if len(sms_message) > SMS_MAX_LENGTH:
                sms_message = sms_message[:SMS_MAX_LENGTH-3] + "..."
            
            # Send SMS
            logger.info(f"Sending SMS to {formatted_recipient} via Twilio")
            sms = client.messages.create(
                body=sms_message,
                from_=TWILIO_PHONE_NUMBER,
                to=formatted_recipient
            )
        except Exception as twilio_error:
            logger.error(f"Twilio authentication or sending error: {str(twilio_error)}")
            
            # Check for common Twilio errors and provide more helpful messages
            error_str = str(twilio_error).lower()
            if "authenticate" in error_str or "auth" in error_str:
                logger.error("Authentication failed. Please check your Twilio credentials (ACCOUNT_SID and AUTH_TOKEN).")
            elif "phone number" in error_str:
                logger.error("Invalid phone number format or permissions issue with the Twilio phone number.")
            
            raise  # Re-raise the exception to be caught by the outer try/except
        
        # Log success with details
        logger.info(f"SMS notification sent to {formatted_recipient}, SID: {sms.sid}")
        logger.info(f"Message content (first 50 chars): {sms_message[:50]}..." if len(sms_message) > 50 else sms_message)
        return True
        
    except TwilioRestException as tre:
        # Handle specific Twilio errors with proper error codes
        error_code = getattr(tre, 'code', None)
        status = getattr(tre, 'status', None)
        
        if error_code == 20003 or status == 401:
            logger.error(f"Twilio authentication failed (Error {error_code}): Check your Account SID and Auth Token")
        elif error_code == 21211:
            logger.error(f"Invalid phone number format (Error {error_code}): {formatted_recipient}")
        elif error_code == 21608:
            logger.error(f"Twilio phone number not enabled for SMS (Error {error_code}): {TWILIO_PHONE_NUMBER}")
        elif error_code == 21610:
            logger.error(f"Message body is too long (Error {error_code})")
        elif error_code == 21612:
            logger.error(f"Twilio account is suspended or closed (Error {error_code})")
        elif error_code == 21614:
            logger.error(f"Invalid 'from' number (Error {error_code}): {TWILIO_PHONE_NUMBER}")
        else:
            logger.error(f"Twilio error (Code {error_code}, Status {status}): {str(tre)}")
            
        # Log the error message for debugging
        logger.debug(f"Full Twilio error: {tre}")
        
        return False
        
    except TwilioException as te:
        # General Twilio exceptions
        logger.error(f"Twilio exception: {str(te)}")
        logger.debug(f"Full Twilio exception details: {te}")
        return False
        
    except Exception as e:
        # Generic exceptions (network issues, etc.)
        logger.error(f"Failed to send SMS: {e}")
        
        # Check if it's a connection error and provide helpful message
        error_str = str(e).lower()
        if "connection" in error_str:
            logger.error("Network connection issue. Please check your internet connection.")
        elif "quota" in error_str or "limit" in error_str:
            logger.error("You may have reached your Twilio messaging quota or rate limit.")
        
        # Log stack trace for debugging
        logger.debug(f"Exception details:", exc_info=True)
        
        # Attempt to fall back to alternative notification method
        logger.warning("SMS sending failed. Consider checking notification settings or using an alternative method.")
        
        return False

def send_sms_notification(db, user_id, message, camera_id=None):
    """
    Send SMS notification to the user
    If camera_id is provided, try to use camera-specific number first
    
    Args:
        db: Firestore database instance
        user_id: ID of the user to send notification to
        message: Message text or detection data dictionary
        camera_id: Optional camera ID to check for camera-specific number
        
    Returns:
        bool: True if SMS was sent successfully
    """
    try:
        # First try to get camera-specific number if camera_id is provided
        phone_number = None
        country_code = None
        
        # Get user notification preferences for country code
        user_prefs = None
        try:
            doc_ref = db.collection('settings').document(f"notifications_{user_id}")
            doc = doc_ref.get()
            if doc.exists:
                user_prefs = doc.to_dict()
                country_code = user_prefs.get('sms', {}).get('country_code')
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
        
        if camera_id:
            try:
                camera_doc = db.collection('cameras').document(camera_id).get()
                if camera_doc.exists:
                    camera_data = camera_doc.to_dict()
                    if camera_data.get('mobile_number'):
                        phone_number = camera_data.get('mobile_number')
            except Exception as e:
                logger.error(f"Error getting camera data: {e}")
        
        # If no camera-specific number, use the user's default
        if not phone_number and user_prefs:
            if user_prefs.get('sms', {}).get('enabled'):
                phone_number = user_prefs.get('sms', {}).get('recipient')
            else:
                logger.info(f"SMS notifications not enabled for user {user_id}")
                return False
        
        if not phone_number:
            logger.warning(f"No phone number available for user {user_id}")
            return False
            
        # Process message
        if isinstance(message, dict):
            # This is detection data
            return send_sms(recipient=phone_number, detection_data=message, country_code=country_code)
        else:
            # This is a plain message
            return send_sms(recipient=phone_number, message=message, country_code=country_code)
            
    except Exception as e:
        logger.error(f"Error sending SMS notification: {e}")
        return False

# These functions are kept for backward compatibility with existing code
def get_available_carriers():
    """
    Return an empty dictionary for carrier selection.
    This function is kept for backward compatibility but is no longer used.
    """
    return {}

def save_carrier_preference(db, user_id: str, carrier: str) -> bool:
    """
    This function is kept for backward compatibility but no longer saves carriers.
    
    Args:
        db: Firestore database instance
        user_id: User ID to save preference for
        carrier: Selected carrier
        
    Returns:
        success: Always returns True
    """
    logger.info(f"Carrier preferences are no longer used. Using Twilio directly.")
    return True