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
        # Create warning document
        warning_ref = db.collection('warnings').document()
        warning_id = warning_ref.id
        
        # Create warning record
        warning_record = {
            'warning_id': warning_id,
            'detection_id': detection_data.get('detection_id', ''),
            'camera_id': detection_data.get('camera_id', ''),
            'camera_name': detection_data.get('camera_name', ''),
            'type': detection_data.get('detection_label', 'unknown'),
            'message': f"{detection_data.get('detection_label', 'Animal')} detected at {detection_data.get('camera_name', 'unknown location')}",
            'screenshot_url': detection_data.get('screenshot_url', ''),
            'google_maps_link': detection_data.get('google_maps_link', ''),
            'severity': determine_severity(detection_data.get('detection_label', '')),
            'timestamp': datetime.now(),
            'active': True,
            'acknowledged': False,
            'notification_status': {
                'email': False,
                'sms': False,
                'telegram': False
            },
            'mobile_number': detection_data.get('mobile_number', '')
        }
        
        # Save warning to database
        warning_ref.set(warning_record)
        
        # Send notifications based on preferences
        notifications_sent = send_notifications(warning_record, notification_preferences)
        
        # Update notification status
        warning_ref.update({
            'notification_status': notifications_sent
        })
        
        logger.info(f"Created warning {warning_id} for {detection_data.get('detection_label', 'unknown')} detection")
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
        'telegram': False
    }
    
    # Construct the message
    message = f"🚨 WildEye Alert: {warning['type']} detected at {warning['camera_name']}\n\n"
    message += f"Severity: {warning['severity'].upper()}\n"
    message += f"Time: {warning['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    if warning['google_maps_link']:
        message += f"Location: {warning['google_maps_link']}\n"
    
    if warning['screenshot_url']:
        message += f"Screenshot: {warning['screenshot_url']}\n"
    
    # Send email if enabled
    if preferences.get('email', {}).get('enabled', False):
        try:
            email_recipient = preferences.get('email', {}).get('recipient')
            if email_recipient:
                notification_status['email'] = send_email(
                    recipient=email_recipient,
                    subject=f"WildEye Alert: {warning['type']} Detected",
                    body=message,
                    screenshot_url=warning.get('screenshot_url')
                )
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    # Send SMS if enabled
    if preferences.get('sms', {}).get('enabled', False):
        try:
            # Use camera's mobile number if available, otherwise use default
            mobile_number = warning.get('mobile_number') or preferences.get('sms', {}).get('recipient')
            if mobile_number:
                notification_status['sms'] = send_sms(
                    recipient=mobile_number,
                    message=message
                )
        except Exception as e:
            logger.error(f"Error sending SMS notification: {e}")
    
    # Send Telegram if enabled
    if preferences.get('telegram', {}).get('enabled', False):
        try:
            chat_id = preferences.get('telegram', {}).get('chat_id', TELEGRAM_CHAT_ID)
            notification_status['telegram'] = send_telegram(
                chat_id=chat_id,
                message=message,
                screenshot_url=warning.get('screenshot_url')
            )
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
    
    return notification_status

def send_email(recipient: str, subject: str, body: str, screenshot_url: Optional[str] = None) -> bool:
    """
    Send an email notification.
    
    Args:
        recipient: Email recipient
        subject: Email subject
        body: Email body text
        screenshot_url: Optional URL to screenshot
        
    Returns:
        success: True if email was sent successfully
    """
    try:
        if not all([EMAIL_USERNAME, EMAIL_PASSWORD, recipient]):
            logger.warning("Missing email credentials or recipient")
            return False
        
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USERNAME
        msg['To'] = recipient
        msg['Subject'] = subject
        
        # Add text body
        msg.attach(MIMEText(body, 'plain'))
        
        # Add HTML body with image if screenshot_url is available
        if screenshot_url:
            html_body = f"""
            <html>
              <body>
                <p>{body.replace('\n', '<br>')}</p>
                <img src="{screenshot_url}" alt="Detection Screenshot" style="max-width:600px">
              </body>
            </html>
            """
            msg.attach(MIMEText(html_body, 'html'))
        
        # Connect to server and send email
        with smtplib.SMTP(EMAIL_SERVER, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"Email notification sent to {recipient}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False

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

def send_telegram(chat_id: str, message: str, screenshot_url: Optional[str] = None) -> bool:
    """
    Send a Telegram notification.
    
    Args:
        chat_id: Telegram chat ID
        message: Message text
        screenshot_url: Optional URL to screenshot
        
    Returns:
        success: True if Telegram message was sent successfully
    """
    try:
        if not all([TELEGRAM_BOT_TOKEN, chat_id]):
            logger.warning("Missing Telegram credentials")
            return False
        
        # Base URL for Telegram Bot API
        base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
        
        # Send text message first
        response = requests.post(
            f"{base_url}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
        )
        response.raise_for_status()
        
        # Send photo if available
        if screenshot_url:
            photo_response = requests.post(
                f"{base_url}/sendPhoto",
                json={
                    "chat_id": chat_id,
                    "photo": screenshot_url,
                    "caption": "Detection Screenshot"
                }
            )
            photo_response.raise_for_status()
        
        logger.info(f"Telegram notification sent to chat {chat_id}")
        return True
        
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
        warning_ref = db.collection('warnings').document(warning_id)
        warning_ref.update({
            'acknowledged': True,
            'acknowledged_at': datetime.now()
        })
        logger.info(f"Warning {warning_id} acknowledged")
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
        warning_ref = db.collection('warnings').document(warning_id)
        warning_ref.update({
            'active': False,
            'resolved_at': datetime.now()
        })
        logger.info(f"Warning {warning_id} resolved")
        return True
        
    except Exception as e:
        logger.error(f"Error resolving warning: {e}")
        return False

def get_notification_preferences(db, user_id: str = None) -> Dict:
    """
    Get notification preferences for a user or global settings.
    
    Args:
        db: Firestore database instance
        user_id: Optional user ID (if None, get global settings)
        
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
            }
        }
        
        # If no user ID, get global settings
        doc_ref = db.collection('settings').document('notifications')
        doc = doc_ref.get()
        
        if doc.exists:
            return doc.to_dict()
        else:
            # Create default settings
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
        user_id: Optional user ID (if None, update global settings)
        
    Returns:
        success: True if preferences were successfully updated
    """
    try:
        # If no user ID, update global settings
        doc_ref = db.collection('settings').document('notifications')
        doc_ref.set(preferences, merge=True)
        logger.info("Notification preferences updated")
        return True
            
    except Exception as e:
        logger.error(f"Error updating notification preferences: {e}")
        return False

def get_all_warnings(db, active_only: bool = False, limit: int = 100) -> List[Dict]:
    """
    Get all warnings from the database.
    
    Args:
        db: Firestore database instance
        active_only: If True, only return active warnings
        limit: Maximum number of warnings to return
        
    Returns:
        warnings: List of warning dictionaries
    """
    try:
        query = db.collection('warnings').order_by('timestamp', direction='desc')
        
        if active_only:
            query = query.where('active', '==', True)
            
        query = query.limit(limit)
        
        return [doc.to_dict() for doc in query.stream()]
            
    except Exception as e:
        logger.error(f"Error getting warnings: {e}")
        return []

# Function to test notification channels
def test_notification_channels(db, preferences: Dict) -> Dict:
    """
    Send test notifications to verify channels are working.
    
    Args:
        db: Firestore database instance
        preferences: Dictionary with notification preferences
        
    Returns:
        status: Dictionary with test results
    """
    test_message = "This is a test notification from WildEye. If you're receiving this, your notification setup is working correctly."
    
    test_warning = {
        'type': 'Test Alert',
        'camera_name': 'Test Camera',
        'severity': 'low',
        'timestamp': datetime.now(),
        'google_maps_link': '',
        'screenshot_url': '',
        'mobile_number': preferences.get('sms', {}).get('recipient', '')
    }
    
    return send_notifications(test_warning, preferences)