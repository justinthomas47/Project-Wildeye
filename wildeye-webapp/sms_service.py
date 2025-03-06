# sms_service.py
import logging
import smtplib
import os
from email.message import EmailMessage
from datetime import datetime
from typing import Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables or config
EMAIL_SERVER = os.environ.get('EMAIL_SERVER', 'smtp.gmail.com')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', 587))
EMAIL_USERNAME = os.environ.get('EMAIL_USERNAME', '')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', '')

# SMS Gateway carrier mappings
SMS_GATEWAYS = {
    # Indian Carriers
    'jio': '@jiomail.com',
    'airtel': '@airtelkk.com',
    'vodafoneidea': '@vimail.in',
    'bsnl': '@bsnlmobile.in',
    'mtnl': '@mtnlmail.in',
    # US Carriers
    'verizon': '@vtext.com',
    'tmobile': '@tmomail.net',
    'sprint': '@messaging.sprintpcs.com',
    'at&t': '@txt.att.net',
    'boost': '@sms.myboostmobile.com',
    'cricket': '@sms.cricketwireless.net',
    'uscellular': '@email.uscc.net',
    # International carriers
    'vodafone': '@vodafone.net',
    'orange': '@orange.net',
    # Default to Airtel if carrier not specified
    'default': '@airtelkk.com'
}

# Animal-specific SMS content - optimized for SMS character limits
ANIMAL_SMS_TEMPLATES = {
    'default': 'Animal detected. Exercise caution.',
    'tiger': 'DANGER! Tiger detected. Stay inside, secure doors/windows, call forest dept immediately.',
    'leopard': 'DANGER! Leopard detected. Stay inside, secure doors/windows, call forest dept immediately.',
    'elephant': 'WARNING! Elephant detected. Stay inside, remain quiet, avoid sudden movements/sounds.',
    'bear': 'DANGER! Bear detected. Stay inside, secure food sources, call forest dept immediately.',
    'wild boar': 'CAUTION! Wild boar detected. Stay inside, keep pets/children away.',
    'wild buffalo': 'DANGER! Wild buffalo detected. Stay inside, extremely unpredictable. Call forest dept.',
    'lion': 'EXTREME DANGER! Lion detected. Stay inside, secure all doors/windows, call emergency services!'
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

def get_animal_sms_content(animal_type: str) -> str:
    """
    Get the appropriate SMS content for the specified animal type.
    
    Args:
        animal_type: The type of animal detected
        
    Returns:
        str: SMS content for the animal
    """
    animal_type = animal_type.lower()
    
    # Check for specific animal matches
    for animal in ANIMAL_SMS_TEMPLATES:
        if animal in animal_type:
            return ANIMAL_SMS_TEMPLATES[animal]
    
    # Return default content if no match
    return ANIMAL_SMS_TEMPLATES['default']

def send_sms_via_email(phone_number: str, detection_data: Dict, carrier: str = 'default') -> bool:
    """
    Send an SMS notification using email-to-SMS gateway.
    
    Args:
        phone_number: Recipient's phone number (can include non-digits like +)
        detection_data: Data about the animal detection
        carrier: Mobile carrier name (verizon, tmobile, etc.)
        
    Returns:
        success: True if SMS was sent successfully
    """
    try:
        if not all([EMAIL_USERNAME, EMAIL_PASSWORD, phone_number]):
            logger.warning("Missing email credentials or recipient phone number")
            return False
        
        # Clean phone number (remove non-digits)
        clean_number = ''.join(filter(str.isdigit, phone_number))
        if not clean_number:
            logger.error(f"Invalid phone number: {phone_number}")
            return False
        
        # Get appropriate gateway suffix
        gateway_suffix = SMS_GATEWAYS.get(carrier.lower(), SMS_GATEWAYS['default'])
        gateway_address = f"{clean_number}{gateway_suffix}"
        
        # Get animal type and appropriate message - FIXED TO CHECK BOTH FIELD NAMES
        # First try 'detection_label' (from detection_handler.py)
        animal_type = detection_data.get('detection_label', '')
        
        # If empty, try 'type' (from warning_system.py)
        if not animal_type:
            animal_type = detection_data.get('type', '')
            
        # If still empty, use default
        if not animal_type:
            animal_type = 'unknown'
            
        sms_content = get_animal_sms_content(animal_type)
        
        # Format timestamp for the SMS
        timestamp_str = ""
        
        # Check for formatted_date first (new field)
        if 'formatted_date' in detection_data:
            timestamp_str = detection_data['formatted_date']
        else:
            # Use timestamp or current time 
            timestamp = detection_data.get('timestamp', datetime.now())
            if isinstance(timestamp, datetime):
                timestamp_str = format_date_dmy(timestamp)
            else:
                timestamp_str = str(timestamp)
        
        # Add detection location and time
        camera_name = detection_data.get('camera_name', 'Unknown location')
        sms_content = f"WildEye ({timestamp_str}): {sms_content} Location: {camera_name}"
        
        # Truncate if too long for SMS
        if len(sms_content) > 160:
            sms_content = sms_content[:157] + "..."
        
        # Create message
        msg = EmailMessage()
        msg.set_content(sms_content)
        msg['From'] = EMAIL_USERNAME
        msg['To'] = gateway_address
        msg['Subject'] = 'WildEye Alert'  # Many carriers ignore subject in SMS gateway
        
        # Connect to server and send email
        with smtplib.SMTP(EMAIL_SERVER, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"SMS notification about {animal_type} sent to {phone_number} via {carrier} gateway")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send SMS via email gateway: {e}")
        return False

def get_available_carriers():
    """
    Get a list of available SMS carriers for the settings page.
    
    Returns:
        List of carrier dictionaries with name and value
    """
    return [
        {'name': 'Reliance Jio', 'value': 'jio'},
        {'name': 'Airtel', 'value': 'airtel'},
        {'name': 'Vodafone Idea (Vi)', 'value': 'vodafoneidea'},
        {'name': 'BSNL', 'value': 'bsnl'},
        {'name': 'MTNL', 'value': 'mtnl'},
        {'name': 'AT&T', 'value': 'at&t'},
        {'name': 'Verizon', 'value': 'verizon'},
        {'name': 'T-Mobile', 'value': 'tmobile'},
        {'name': 'Sprint', 'value': 'sprint'},
        {'name': 'Boost Mobile', 'value': 'boost'},
        {'name': 'Cricket', 'value': 'cricket'},
        {'name': 'US Cellular', 'value': 'uscellular'},
        {'name': 'Vodafone', 'value': 'vodafone'},
        {'name': 'Orange', 'value': 'orange'},
        {'name': 'Other/Unknown', 'value': 'default'}
    ]

def save_carrier_preference(db, user_id: str, carrier: str):
    """
    Save user's mobile carrier preference to the database.
    
    Args:
        db: Firestore database instance
        user_id: User ID
        carrier: Mobile carrier name
        
    Returns:
        success: True if preference was saved successfully
    """
    try:
        doc_ref = db.collection('settings').document(f'notifications_{user_id}')
        doc_ref.update({
            'sms.carrier': carrier.lower()
        })
        return True
    except Exception as e:
        logger.error(f"Error saving carrier preference: {e}")
        return False

def load_carrier_preference(db, user_id: str) -> str:
    """
    Load user's mobile carrier preference from the database.
    
    Args:
        db: Firestore database instance
        user_id: User ID
        
    Returns:
        carrier: Mobile carrier name or 'default' if not set
    """
    try:
        doc_ref = db.collection('settings').document(f'notifications_{user_id}')
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            return data.get('sms', {}).get('carrier', 'default')
        return 'default'
    except Exception as e:
        logger.error(f"Error loading carrier preference: {e}")
        return 'default'