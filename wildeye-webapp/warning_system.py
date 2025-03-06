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
        formatted_time = current_time.strftime("%Y-%m-%d %I:%M:%S %p")
        
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
            'formatted_timestamp': formatted_time,      # Add formatted time string
            'active': True,
            'acknowledged': False,
            'notification_status': {
                'email': False,
                'sms': False,
                'telegram': False
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
        
        logger.info(f"Created warning {warning_id} for {detection_data.get('detection_label', 'unknown')} detection at {formatted_time}")
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
    
    # Construct the message - Use formatted_timestamp if available
    message = f"🚨 WildEye Alert: {warning['type']} detected at {warning['camera_name']}\n\n"
    message += f"Severity: {warning['severity'].upper()}\n"
    
    # Use formatted timestamp if available, otherwise format the timestamp
    if 'formatted_timestamp' in warning:
        message += f"Time: {warning['formatted_timestamp']}\n"
    else:
        message += f"Time: {warning['timestamp'].strftime('%Y-%m-%d %I:%M:%S %p')}\n"
    
    # Add location information without the direct link
    if warning['google_maps_link']:
        message += f"Location: {warning['camera_name']}\n"
    
    # Mention that screenshot is available but don't include the URL
    if warning['screenshot_url']:
        message += f"Screenshot is available in email or app\n"
    
    # Send email if enabled
    if preferences.get('email', {}).get('enabled', False):
        try:
            email_recipient = preferences.get('email', {}).get('recipient')
            if email_recipient:
                notification_status['email'] = send_email(
                    recipient=email_recipient,
                    subject=f"WildEye Alert: {warning['type']} Detected",
                    body=message,
                    screenshot_url=warning.get('screenshot_url'),
                    warning=warning
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

def send_email(recipient: str, subject: str, body: str, screenshot_url: Optional[str] = None, warning: Dict = None) -> bool:
    """
    Send an enhanced email notification with location details, animal photo and precaution measures.
    
    Args:
        recipient: Email recipient
        subject: Email subject
        body: Basic email body text
        screenshot_url: Optional URL to screenshot
        warning: Warning information dictionary
        
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
        
        # Add text version as fallback
        msg.attach(MIMEText(body, 'plain'))
        
        # Get animal type for precaution recommendations
        animal_type = warning.get('type', '').lower() if warning else 'unknown'
        precautions = get_precautions_for_animal(animal_type)
        
        # Format location information
        location_info = ""
        if warning and warning.get('google_maps_link'):
            location_info = f"""
            <div style="margin: 25px 0; background-color: #f5f5f5; border-radius: 6px; padding: 20px; border-left: 5px solid #2ecc71;">
                <h3 style="color: #2c3e50; margin-top: 0; margin-bottom: 15px;">📍 Location Details</h3>
                <p>The detection occurred at <strong>{warning.get('camera_name', 'Unknown location')}</strong>.</p>
                <p style="margin-bottom: 0;"><a href="{warning['google_maps_link']}" style="background-color: #2ecc71; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; display: inline-block; font-weight: bold;">View on Google Maps</a></p>
            </div>
            """
        
        # Get formatted timestamp for email
        if warning:
            if 'formatted_timestamp' in warning:
                timestamp_display = warning['formatted_timestamp']
            else:
                timestamp_display = warning.get('timestamp').strftime('%Y-%m-%d %I:%M:%S %p') if warning.get('timestamp') else 'Unknown'
        else:
            timestamp_display = 'Unknown'
            
        # Create HTML email with enhanced layout
        html_body = f"""
        <html>
          <head>
            <style>
              body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; }}
              .container {{ border: 1px solid #ddd; border-radius: 8px; padding: 30px; background-color: #fcfcfc; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
              .header {{ background-color: #2c3e50; padding: 20px; border-radius: 6px; margin-bottom: 25px; color: white; }}
              .severity-high {{ color: #721c24; background-color: #f8d7da; padding: 15px; border-radius: 6px; border-left: 5px solid #dc3545; }}
              .severity-medium {{ color: #856404; background-color: #fff3cd; padding: 15px; border-radius: 6px; border-left: 5px solid #ffc107; }}
              .severity-low {{ color: #0c5460; background-color: #d1ecf1; padding: 15px; border-radius: 6px; border-left: 5px solid #17a2b8; }}
              .image-container {{ margin: 25px 0; }}
              .image-container img {{ max-width: 100%; height: auto; border-radius: 6px; }}
              .precautions {{ background-color: #e8f4f8; padding: 20px; border-radius: 6px; margin-top: 25px; border-left: 5px solid #3498db; }}
              .precautions ul {{ padding-left: 20px; }}
              .precautions li {{ margin-bottom: 8px; }}
              .footer {{ font-size: 13px; text-align: center; margin-top: 35px; color: #6c757d; padding-top: 15px; border-top: 1px solid #eee; }}
            </style>
          </head>
          <body>
            <div class="container">
              <div class="header">
                <h2 style="margin-top: 0; margin-bottom: 10px;">🚨 WildEye Wildlife Detection Alert</h2>
                <p style="margin-bottom: 0; opacity: 0.9;">This automated alert has been generated by the WildEye wildlife monitoring system.</p>
              </div>
              
              <div class="severity-{warning.get('severity', 'low') if warning else 'low'}">
                <h3 style="margin-top: 0;">Alert Details</h3>
                <p><strong>Animal Detected:</strong> {warning.get('type', 'Unknown animal').title() if warning else 'Unknown'}</p>
                <p><strong>Time:</strong> {timestamp_display}</p>
                <p style="margin-bottom: 0;"><strong>Severity:</strong> {warning.get('severity', 'Unknown').upper() if warning else 'Unknown'}</p>
              </div>
              
              {location_info}
              
              <div class="image-container">
                <h3 style="color: #2c3e50;">📸 Detection Image</h3>
                {f'''
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); background-color: #fff;">
                  <a href="{screenshot_url}" style="display: block;">
                    <img src="{screenshot_url}" alt="Detection Screenshot" style="max-width:100%; border-radius: 4px; display: block; margin: 0 auto;">
                    <div style="text-align: center; margin-top: 10px;">
                      <span style="background-color: #3498db; color: white; padding: 6px 12px; text-decoration: none; border-radius: 4px; display: inline-block;">View Full Image</span>
                    </div>
                  </a>
                </div>
                ''' if screenshot_url else '<p>No image available for this detection.</p>'}
              </div>
              
              <div class="precautions">
                <h3 style="color: #2c3e50;">⚠️ Recommended Precautions</h3>
                <ul>
                  {precautions}
                </ul>
              </div>
              
              <div class="footer">
                <p>This is an automated message from the WildEye system. Please do not reply to this email.</p>
                <p>To manage your notification settings, please log in to your WildEye account.</p>
              </div>
            </div>
          </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, 'html'))
        
        # Connect to server and send email
        with smtplib.SMTP(EMAIL_SERVER, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"Enhanced email notification sent to {recipient}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False

def get_precautions_for_animal(animal_type: str) -> str:
    """
    Get recommended precautions based on animal type.
    
    Args:
        animal_type: Type of animal detected
        
    Returns:
        HTML formatted list of precautions
    """
    # Default precautions for all wildlife
    default_precautions = [
        "Keep a safe distance from the animal and avoid approaching it",
        "Ensure children and pets remain indoors or supervised",
        "Do not attempt to feed or interact with the animal",
        "Report the sighting to local wildlife authorities if the animal appears injured or in distress"
    ]
    
    # Animal-specific precautions
    precautions_map = {
        'tiger': [
            "Evacuate the area immediately and move to a secure location",
            "Alert all staff and residents in the vicinity",
            "Contact forest department or wildlife rangers urgently",
            "Do not run if you see the tiger - back away slowly while facing it",
            "Avoid outdoor activities until authorities confirm it's safe"
        ],
        'leopard': [
            "Stay indoors and secure all entry points to buildings",
            "Keep livestock in protected enclosures",
            "Travel in groups and carry noise-making devices if movement is necessary",
            "Contact wildlife authorities immediately",
            "Be particularly cautious during dawn and dusk hours"
        ],
        'elephant': [
            "Maintain a minimum distance of 100 meters",
            "Never position yourself between a mother and her calf",
            "If in a vehicle, turn off the engine and remain quiet",
            "Avoid sudden movements or loud noises",
            "Be aware that elephants can charge with little warning"
        ],
        'bear': [
            "Make noise to avoid surprising the bear",
            "Secure all food and garbage in bear-proof containers",
            "If you encounter a bear, speak calmly and back away slowly",
            "Never run from a bear - this may trigger a chase response",
            "If attacked, use bear spray if available"
        ],
        'wild boar': [
            "Keep distance as wild boars can be aggressive",
            "Secure gardens and crops with appropriate fencing",
            "Avoid walking dogs in areas where wild boars have been sighted",
            "Never corner or threaten a wild boar"
        ],
        'deer': [
            "Drive cautiously in the area, especially at dawn and dusk",
            "Keep dogs leashed in areas with deer",
            "Do not approach fawns, even if they appear abandoned"
        ],
        'wolf': [
            "Keep pets indoors or on short leashes",
            "Never approach or follow wolves",
            "Eliminate potential food sources near human settlements",
            "If you encounter a wolf, make yourself look large and make loud noises"
        ],
        'hyena': [
            "Secure livestock in predator-proof enclosures",
            "Do not leave food waste accessible",
            "Keep children close and supervised in areas with reported hyena activity",
            "If encountered, maintain eye contact and back away slowly"
        ]
    }
    
    # Get specific precautions or use defaults
    specific_precautions = precautions_map.get(animal_type, default_precautions)
    
    # Format as HTML list items
    formatted_precautions = '\n'.join([f'<li>{precaution}</li>' for precaution in specific_precautions])
    
    return formatted_precautions

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
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %I:%M:%S %p")
        
        warning_ref = db.collection('warnings').document(warning_id)
        warning_ref.update({
            'acknowledged': True,
            'acknowledged_at': current_time,
            'acknowledged_at_formatted': formatted_time
        })
        logger.info(f"Warning {warning_id} acknowledged at {formatted_time}")
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
        formatted_time = current_time.strftime("%Y-%m-%d %I:%M:%S %p")
        
        warning_ref = db.collection('warnings').document(warning_id)
        warning_ref.update({
            'active': False,
            'resolved_at': current_time,
            'resolved_at_formatted': formatted_time
        })
        logger.info(f"Warning {warning_id} resolved at {formatted_time}")
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
            return doc.to_dict()
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
        
        test_message = f"This is a test notification from WildEye at {formatted_time}. If you're receiving this, your notification setup is working correctly."
        
        test_warning = {
            'type': 'Test Alert',
            'camera_name': 'Test Camera',
            'severity': 'low',
            'timestamp': current_time,
            'formatted_timestamp': formatted_time,
            'google_maps_link': '',
            'screenshot_url': '',
            'mobile_number': preferences.get('sms', {}).get('recipient', '')
        }
        
        notification_results = send_notifications(test_warning, preferences)
        
        return {
            'success': True,
            'email': notification_results['email'],
            'sms': notification_results['sms'],
            'telegram': notification_results['telegram'],
            'message': 'Test notifications sent'
        }
    except Exception as e:
        logger.error(f"Error testing notification channels: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to test notifications'
        }