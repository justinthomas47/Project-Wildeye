# warning_system.py
import logging
import smtplib
import requests
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client
from datetime import datetime
from typing import Dict, Optional, List
from telegram_service import send_telegram_notification
from email_service import send_email  # Import the enhanced email function
from sms_service import send_sms as send_sms_service
from call_service import make_call, should_call_for_detection  # Import the call service

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables or config
EMAIL_SERVER = os.environ.get('EMAIL_SERVER', 'smtp.gmail.com')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', 587))
EMAIL_USERNAME = os.environ.get('EMAIL_USERNAME', '')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', '')

# Twilio credentials
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', '')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER', '')

# Telegram credentials
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

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

def create_warning(db, detection_data: Dict, notification_preferences: Dict) -> str:
    """
    Create a warning in the database based on a detection.
    
    Args:
        db: Firestore database instance
        detection_data: Dictionary containing detection information
        notification_preferences: Dictionary with notification preferences
        
    Returns:
        warning_id: The ID of the created warning
    """
    try:
        # Get the camera details to find the owner
        camera_id = detection_data.get('camera_id', '')
        owner_uid = None
        
        if camera_id:
            camera_ref = db.collection('cameras').document(camera_id)
            camera_doc = camera_ref.get()
            if camera_doc.exists:
                camera_data = camera_doc.to_dict()
                owner_uid = camera_data.get('owner_uid')
        
        # Create warning document
        warning_ref = db.collection('warnings').document()
        warning_id = warning_ref.id
        
        # Get animal type, ensuring field name consistency
        detection_label = detection_data.get('detection_label', 'unknown')
        
        # Current time
        current_time = datetime.now()
        
        # Format timestamp for display in 12-hour format
        formatted_time = current_time.strftime("%Y-%m-%d %I:%M:%S %p")  # Keep original for backwards compatibility
        formatted_date = format_date_dmy(current_time)  # DD-MM-YYYY format
        
        # Create warning record
        warning_record = {
            'warning_id': warning_id,
            'detection_id': detection_data.get('detection_id', ''),
            'camera_id': camera_id,
            'camera_name': detection_data.get('camera_name', ''),
            'type': detection_label,                    # Keep 'type' for backward compatibility
            'detection_label': detection_label,         # Add 'detection_label' to ensure consistency
            'message': f"{detection_label} detected at {detection_data.get('camera_name', 'unknown location')}",
            'screenshot_url': detection_data.get('screenshot_url', ''),
            'google_maps_link': detection_data.get('google_maps_link', ''),
            'severity': determine_severity(detection_label),
            'timestamp': current_time,                  # Keep datetime object for sorting
            'formatted_timestamp': formatted_time,      # Keep original format for backwards compatibility
            'formatted_date': formatted_date,           # Add new day-month-year format
            'active': True,
            'acknowledged': False,
            'notification_status': {
                'email': False,
                'sms': False,
                'telegram': False,
                'call': False                           # Add call status
            },
            'mobile_number': detection_data.get('mobile_number', ''),
            'owner_uid': owner_uid  # Add owner UID to link warning to user
        }
        
        # Save warning to database
        warning_ref.set(warning_record)
        
        # Get user-specific notification preferences if we have owner_uid
        if owner_uid:
            user_preferences = get_notification_preferences(db, owner_uid)
            # Use user preferences if they exist, otherwise use provided preferences
            if user_preferences:
                notification_preferences = user_preferences
        
        # Send notifications based on preferences
        notifications_sent = send_notifications(warning_record, notification_preferences)
        
        # Update notification status
        warning_ref.update({
            'notification_status': notifications_sent
        })
        
        logger.info(f"Created warning {warning_id} for {detection_data.get('detection_label', 'unknown')} detection at {formatted_date}")
        return warning_id
        
    except Exception as e:
        logger.error(f"Error creating warning: {e}")
        return None

def determine_severity(detection_label: str) -> str:
    """
    Determine the severity of the warning based on the detection label.
    
    Args:
        detection_label: Type of animal detected
        
    Returns:
        severity: 'high', 'medium', or 'low'
    """
    # Define high-priority animals (customize this list)
    high_priority = ['tiger', 'leopard', 'elephant', 'bear']
    
    # Define medium-priority animals
    medium_priority = ['wild boar', 'deer', 'wolf', 'hyena']
    
    # Convert to lowercase for comparison
    detection_label = detection_label.lower()
    
    if any(animal in detection_label for animal in high_priority):
        return 'high'
    elif any(animal in detection_label for animal in medium_priority):
        return 'medium'
    else:
        return 'low'

def send_notifications(warning: Dict, preferences: Dict) -> Dict:
    """
    Send notifications about the warning through enabled channels.
    
    Args:
        warning: Warning information
        preferences: Dictionary with notification preferences
        
    Returns:
        status: Dictionary with notification statuses
    """
    notification_status = {
        'email': False,
        'sms': False,
        'telegram': False,
        'call': False  # Add call status to notification status
    }
    
    # Use formatted_date if available, otherwise format the timestamp
    if 'formatted_date' in warning:
        time_display = warning['formatted_date']
    else:
        time_display = warning['formatted_timestamp'] if 'formatted_timestamp' in warning else format_date_dmy(warning['timestamp'])
    
    # Construct the message
    message = f"🚨 WildEye Alert: {warning['type']} detected at {warning['camera_name']}\n\n"
    message += f"Severity: {warning['severity'].upper()}\n"
    message += f"Time: {time_display}\n"
    
    # Add location information without the direct link
    if warning['google_maps_link']:
        message += f"Location: {warning['camera_name']}\n"
    
    # Mention that screenshot is available but don't include the URL
    if warning['screenshot_url']:
        message += f"Screenshot is available in email or app\n"
    
    # Send email if enabled - now using the enhanced email_service function
    if preferences.get('email', {}).get('enabled', False):
        try:
            email_recipient = preferences.get('email', {}).get('recipient')
            if email_recipient:
                email_subject = f"WildEye Alert: {warning['type']} Detected"
                # Use the enhanced email function from email_service.py
                notification_status['email'] = send_email(
                    recipient=email_recipient,
                    subject=email_subject,
                    detection_data=warning
                )
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    # Send SMS if enabled
    if preferences.get('sms', {}).get('enabled', False):
        try:
            # Use camera's mobile number if available, otherwise use default
            mobile_number = warning.get('mobile_number') or preferences.get('sms', {}).get('recipient')
            country_code = preferences.get('sms', {}).get('country_code', '+91')  # Default to India code if not specified
            if mobile_number:
                from sms_service import send_sms as send_sms_service
                notification_status['sms'] = send_sms_service(
                    recipient=mobile_number,
                    message=message,
                    country_code=country_code 
                )
        except Exception as e:
            logger.error(f"Error sending SMS notification: {e}")
    
    # Send Telegram if enabled
    if preferences.get('telegram', {}).get('enabled', False):
        try:
            chat_id = preferences.get('telegram', {}).get('chat_id', TELEGRAM_CHAT_ID)
            # Pass the entire warning object to the telegram function
            notification_status['telegram'] = send_telegram(
                chat_id=chat_id,
                message=message,
                screenshot_url=warning.get('screenshot_url'),
                warning=warning  # Pass the entire warning object
            )
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
            
    # Make voice call if enabled and the detection meets the threshold criteria
    if preferences.get('call', {}).get('enabled', False) and should_call_for_detection(warning, preferences):
        try:
            # Use camera's mobile number if available, otherwise use default from call preferences
            mobile_number = warning.get('mobile_number') or preferences.get('call', {}).get('recipient')
            
            # If no call-specific number, try the SMS number as fallback
            if not mobile_number:
                mobile_number = preferences.get('sms', {}).get('recipient')
                
            # Format with country code if needed
            if mobile_number and preferences.get('call', {}).get('country_code'):
                country_code = preferences.get('call', {}).get('country_code')
                # Only add country code if it doesn't already have one
                if not mobile_number.startswith('+'):
                    # Remove leading zeros if present
                    mobile_number = mobile_number.lstrip('0')
                    # Add country code
                    mobile_number = f"{country_code}{mobile_number}"
            
            if mobile_number:
                notification_status['call'] = make_call(
                    recipient=mobile_number,
                    detection_data=warning
                )
        except Exception as e:
            logger.error(f"Error making call notification: {e}")
    
    return notification_status

def send_sms(recipient: str, message: str) -> bool:
    """
    Send an SMS notification using Twilio.
    
    Args:
        recipient: Phone number to send SMS to
        message: Message text
        
    Returns:
        success: True if SMS was sent successfully
    """
    try:
        if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, recipient]):
            logger.warning("Missing Twilio credentials or recipient phone number")
            return False
        
        # Initialize Twilio client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Limit message length for SMS
        if len(message) > 1600:
            message = message[:1597] + "..."
        
        # Send SMS
        sms = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=recipient
        )
        
        logger.info(f"SMS notification sent to {recipient}, SID: {sms.sid}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send SMS: {e}")
        return False

def send_telegram(chat_id: str, message: str, screenshot_url: Optional[str] = None, warning: Dict = None) -> bool:
    """
    Send a Telegram notification.
    
    Args:
        chat_id: Telegram chat ID
        message: Message text
        screenshot_url: Optional URL to screenshot
        warning: Optional warning data dictionary
        
    Returns:
        success: True if Telegram message was sent successfully
    """
    try:
        # Import the telegram notification function
        from telegram_service import send_telegram_notification
        
        if not chat_id:
            logger.warning("Missing Telegram chat ID")
            return False
        
        # If we have a full warning object, use the enhanced notification function
        if warning:
            # Make sure screenshot_url is included in the warning data
            if screenshot_url and 'screenshot_url' not in warning:
                warning['screenshot_url'] = screenshot_url
                
            # Call the enhanced function from telegram_service
            return send_telegram_notification(chat_id, warning)
        
        # Otherwise, create a minimal warning object with just the message
        else:
            minimal_warning = {
                'detection_label': 'Alert',
                'type': 'Alert',
                'camera_name': 'WildEye System',
                'timestamp': datetime.now(),
                'formatted_date': format_date_dmy(datetime.now()),
                'message': message,
                'screenshot_url': screenshot_url
            }
            
            return send_telegram_notification(chat_id, minimal_warning)
            
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")
        return False

def acknowledge_warning(db, warning_id: str) -> bool:
    """
    Mark a warning as acknowledged.
    
    Args:
        db: Firestore database instance
        warning_id: ID of the warning to acknowledge
        
    Returns:
        success: True if warning was successfully acknowledged
    """
    try:
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %I:%M:%S %p")  # Original format
        formatted_date = format_date_dmy(current_time)  # DD-MM-YYYY format
        
        warning_ref = db.collection('warnings').document(warning_id)
        warning_ref.update({
            'acknowledged': True,
            'acknowledged_at': current_time,
            'acknowledged_at_formatted': formatted_time,
            'acknowledged_at_formatted_date': formatted_date
        })
        logger.info(f"Warning {warning_id} acknowledged at {formatted_date}")
        return True
        
    except Exception as e:
        logger.error(f"Error acknowledging warning: {e}")
        return False

def resolve_warning(db, warning_id: str) -> bool:
    """
    Mark a warning as resolved/inactive.
    
    Args:
        db: Firestore database instance
        warning_id: ID of the warning to resolve
        
    Returns:
        success: True if warning was successfully resolved
    """
    try:
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %I:%M:%S %p")  # Original format
        formatted_date = format_date_dmy(current_time)  # DD-MM-YYYY format
        
        warning_ref = db.collection('warnings').document(warning_id)
        warning_ref.update({
            'active': False,
            'resolved_at': current_time,
            'resolved_at_formatted': formatted_time,
            'resolved_at_formatted_date': formatted_date
        })
        logger.info(f"Warning {warning_id} resolved at {formatted_date}")
        return True
        
    except Exception as e:
        logger.error(f"Error resolving warning: {e}")
        return False

def get_notification_preferences(db, user_id: str = None) -> Dict:
    """
    Get notification preferences for a user or global settings.
    
    Args:
        db: Firestore database instance
        user_id: User ID to get preferences for
        
    Returns:
        preferences: Dictionary with notification preferences
    """
    try:
        # Default preferences
        default_preferences = {
            'email': {
                'enabled': False,
                'recipient': ''
            },
            'sms': {
                'enabled': False,
                'recipient': ''
            },
            'telegram': {
                'enabled': False,
                'chat_id': TELEGRAM_CHAT_ID
            },
            'call': {
                'enabled': False,
                'recipient': '',
                'country_code': '+91',  # Default to India
                'threshold': 'high'     # Options: 'high', 'medium', 'all'
            }
        }
        
        # If user_id provided, get user-specific settings
        if user_id:
            doc_ref = db.collection('settings').document(f'notifications_{user_id}')
        else:
            # Fall back to global settings
            doc_ref = db.collection('settings').document('notifications')
            
        doc = doc_ref.get()
        
        if doc.exists:
            preferences = doc.to_dict()
            
            # Ensure call settings exist in older preference records
            if 'call' not in preferences:
                preferences['call'] = default_preferences['call']
                # If SMS is enabled, use that number for calls by default
                if preferences.get('sms', {}).get('enabled', False):
                    preferences['call']['recipient'] = preferences.get('sms', {}).get('recipient', '')
                    preferences['call']['country_code'] = preferences.get('sms', {}).get('country_code', '+91')
                
            return preferences
        else:
            # Create default settings
            if user_id:
                default_preferences['owner_uid'] = user_id
                doc_ref.set(default_preferences)
            return default_preferences
            
    except Exception as e:
        logger.error(f"Error getting notification preferences: {e}")
        return default_preferences

def update_notification_preferences(db, preferences: Dict, user_id: str = None) -> bool:
    """
    Update notification preferences.
    
    Args:
        db: Firestore database instance
        preferences: Dictionary with new preferences
        user_id: User ID to update preferences for
        
    Returns:
        success: True if preferences were successfully updated
    """
    try:
        if user_id:
            # Update user-specific settings
            doc_ref = db.collection('settings').document(f'notifications_{user_id}')
            # Ensure owner_uid field is set
            preferences['owner_uid'] = user_id
        else:
            # Update global settings
            doc_ref = db.collection('settings').document('notifications')
            
        doc_ref.set(preferences, merge=True)
        logger.info("Notification preferences updated")
        return True
            
    except Exception as e:
        logger.error(f"Error updating notification preferences: {e}")
        return False

def get_all_warnings(db, active_only: bool = False, limit: int = 100, user_id: str = None) -> List[Dict]:
    """
    Get all warnings from the database, optionally filtered by user.
    
    Args:
        db: Firestore database instance
        active_only: If True, only return active warnings
        limit: Maximum number of warnings to return
        user_id: If provided, only return warnings for this user
        
    Returns:
        warnings: List of warning dictionaries
    """
    try:
        query = db.collection('warnings').order_by('timestamp', direction='desc')
        
        if active_only:
            query = query.where('active', '==', True)
            
        if user_id:
            query = query.where('owner_uid', '==', user_id)
            
        query = query.limit(limit)
        
        return [doc.to_dict() for doc in query.stream()]
            
    except Exception as e:
        logger.error(f"Error getting warnings: {e}")
        return []

# Function to test notification channels
def test_notification_channels(db, user_id: str) -> Dict:
    """
    Send test notifications to verify channels are working.
    
    Args:
        db: Firestore database instance
        user_id: User ID to test notifications for
        
    Returns:
        status: Dictionary with test results
    """
    try:
        # Get the user's notification preferences
        preferences = get_notification_preferences(db, user_id)
        
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %I:%M:%S %p")
        formatted_date = format_date_dmy(current_time)  # DD-MM-YYYY format
        
        test_message = f"This is a test notification from WildEye at {formatted_date}. If you're receiving this, your notification setup is working correctly."
        
        # For call testing, we'll simulate a medium severity detection
        test_warning = {
            'type': 'Test Alert',
            'detection_label': 'Test Alert',
            'camera_name': 'Test Camera',
            'severity': 'medium',  # Use medium severity for testing
            'timestamp': current_time,
            'formatted_timestamp': formatted_time,
            'formatted_date': formatted_date,
            'google_maps_link': '',
            'screenshot_url': 'https://via.placeholder.com/800x600.png?text=WildEye+Test',
            'mobile_number': preferences.get('sms', {}).get('recipient', '') or preferences.get('call', {}).get('recipient', '')
        }
        
        # Create temporary preferences for testing calls
        # For testing purposes, if call is enabled, we'll force the threshold to 'all'
        # so the test call will be made regardless of the normal threshold setting
        test_preferences = preferences.copy()
        if test_preferences.get('call', {}).get('enabled', False):
            if 'call' not in test_preferences:
                test_preferences['call'] = {}
            test_preferences['call']['threshold'] = 'all'
        
        notification_results = send_notifications(test_warning, test_preferences)
        
        return {
            'success': True,
            'email': notification_results['email'],
            'sms': notification_results['sms'],
            'telegram': notification_results['telegram'],
            'call': notification_results['call'],
            'message': 'Test notifications sent'
        }
    except Exception as e:
        logger.error(f"Error testing notification channels: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to test notifications'
        }