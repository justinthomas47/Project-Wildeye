# warning_system.py
import logging
import smtplib
import requests
import os
import re
import math
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from telegram_service import send_telegram_notification
from email_service import send_email  # Import the enhanced email function
from sms_service import send_sms as send_sms_service
from call_service import make_call, should_call_for_detection  # Import the call service
from distance_calculation import LocationUtils

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

def extract_coordinates_from_maps_link(maps_link):
    """
    Extract coordinates from a Google Maps link using LocationUtils.
    
    Args:
        maps_link: Google Maps URL
        
    Returns:
        tuple: (latitude, longitude) or None if extraction fails
    """
    coords = LocationUtils.extract_coordinates(maps_link)
    if coords:
        return coords
    
    # Fall back to default coordinates if extraction fails
    logger.warning(f"Could not extract coordinates from {maps_link}, using default coordinates")
    return 20.5937, 78.9629  # Default to center of India

def calculate_distance(lat1, lng1, lat2, lng2):
    """
    Calculate the distance between two points on Earth.
    
    Args:
        lat1, lng1: Latitude and longitude of point 1
        lat2, lng2: Latitude and longitude of point 2
        
    Returns:
        distance: Distance in kilometers
    """
    return LocationUtils.calculate_distance((lat1, lng1), (lat2, lng2))

def format_distance(distance_km):
    """
    Format distance in kilometers with proper units.
    
    Args:
        distance_km: Distance in kilometers
        
    Returns:
        formatted_distance: Formatted distance string (number only, no units)
    """
    # Return numeric value only without units for display in template
    return LocationUtils.format_distance_for_display(distance_km, with_units=False)

def create_warning(db, detection_data: Dict, notification_preferences: Dict) -> str:
    """
    Create a warning in the database based on a detection and trigger broadcast alerts.
    
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
            'is_broadcast': False,                      # This is NOT a broadcast alert (it's a direct camera alert)
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
        
        # Send broadcast alerts if this has a location and is a medium or high severity warning
        if warning_record['google_maps_link'] and warning_record['severity'] in ['high', 'medium']:
            logger.info(f"Triggering broadcast alerts for warning {warning_id}")
            broadcast_result = broadcast_warning(db, warning_record)
            
            # Add broadcast info to warning
            warning_ref.update({
                'broadcast_status': broadcast_result
            })
            
            if broadcast_result.get('broadcast'):
                logger.info(f"Broadcast alert sent for warning {warning_id} to {broadcast_result.get('users_notified', 0)} users")
            else:
                logger.warning(f"No broadcast alerts sent for warning {warning_id}: {broadcast_result.get('reason', 'unknown reason')}")
        else:
            reason = "No Google Maps link" if not warning_record['google_maps_link'] else f"Low severity ({warning_record['severity']})"
            logger.info(f"Skipping broadcast for warning {warning_id}: {reason}")
        
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
    
    # Check if this is a broadcast alert
    is_broadcast = warning.get('is_broadcast', False)
    
    # Get distance for broadcast alerts
    distance = None
    if is_broadcast:
        # Use distance_km if available (numeric value)
        if 'distance_km' in warning:
            distance = warning['distance_km']
        # Otherwise use distance (might be pre-formatted)
        elif 'distance' in warning:
            distance = warning['distance']
    
    # Construct the message
    message = f"🚨 WildEye Alert: {warning['type']} detected at {warning['camera_name']}\n\n"
    message += f"Severity: {warning['severity'].upper()}\n"
    message += f"Time: {time_display}\n"
    
    # For broadcast alerts, include distance information
    if is_broadcast and distance is not None:
        message += f"Distance: {distance} km from your location\n"
    
    # Add location information without the direct link
    if warning['google_maps_link']:
        message += f"Location: {warning['camera_name']}\n"
    
    # Mention that screenshot is available but don't include the URL
    if warning['screenshot_url']:
        message += f"Screenshot is available in email or app\n"
    
    # Send email if enabled - now using the enhanced email function
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
                # Pass the entire warning object, including broadcast flag and distance
                notification_status['sms'] = send_sms_service(
                    recipient=mobile_number,
                    message=message if message else None,  # Provide message as a fallback
                    country_code=country_code,
                    detection_data=warning  # Pass the entire warning object
                )
        except Exception as e:
            logger.error(f"Error sending SMS notification: {e}")
    
    # Send Telegram if enabled
    if preferences.get('telegram', {}).get('enabled', False):
        try:
            chat_id = preferences.get('telegram', {}).get('chat_id', TELEGRAM_CHAT_ID)
            logger.info(f"Sending Telegram notification to chat_id: {chat_id} for {'broadcast' if warning.get('is_broadcast') else 'direct'} alert")
            
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
                # Pass the entire warning object to make_call
                notification_status['call'] = make_call(
                    recipient=mobile_number,
                    detection_data=warning  # Pass the entire warning object
                )
        except Exception as e:
            logger.error(f"Error making call notification: {e}")
    
    return notification_status

def broadcast_warning(db, warning):
    """
    Send broadcast alerts to nearby users based on their preferences.
    This function is called after a regular warning is created for the camera owner.
    
    Args:
        db: Firestore database instance
        warning: The original warning to broadcast
        
    Returns:
        dict: Summary of broadcast notifications sent
    """
    try:
        logger.info(f"Processing broadcast for warning {warning.get('warning_id')}")
        
        # Check if warning has location information
        if not warning.get('google_maps_link'):
            logger.info(f"Warning {warning.get('warning_id')} has no location data, skipping broadcast")
            return {'broadcast': False, 'reason': 'no_location'}
        
        # Extract location coordinates using improved method
        from distance_calculation import LocationUtils
        
        warning_location = None
        # Try multiple times with increasing timeouts to extract coordinates
        for attempt in range(3):
            try:
                logger.info(f"Attempting to extract coordinates (attempt {attempt+1}/3)")
                warning_location = LocationUtils.extract_coordinates(warning.get('google_maps_link'))
                if warning_location:
                    break
                time.sleep(1)  # Wait a bit before retrying
            except Exception as e:
                logger.error(f"Error extracting coordinates (attempt {attempt+1}): {e}")
                time.sleep(1)
        
        if not warning_location:
            # Try the specialized method for shortened URLs
            try:
                logger.info("Attempting specialized extraction for shortened URL")
                warning_location = LocationUtils.extract_coordinates_from_shortened_url(warning.get('google_maps_link'))
            except Exception as e:
                logger.error(f"Error with specialized extraction: {e}")
        
        if not warning_location:
            # Last resort - try to fetch the maps link directly
            try:
                logger.info("Attempting direct HTML content extraction")
                import requests
                response = requests.get(warning.get('google_maps_link'), timeout=20)
                content = response.text
                # Look for coordinate patterns in the HTML
                import re
                
                # Try multiple patterns
                patterns = [
                    r'@(-?\d+\.\d+),(-?\d+\.\d+)',
                    r'!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)'
                ]
                
                for pattern in patterns:
                    matches = re.search(pattern, content)
                    if matches:
                        if pattern == r'!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)':
                            # For !3d!4d pattern, order is lat then lng
                            lat = float(matches.group(1))
                            lng = float(matches.group(2))
                        else:
                            # For other patterns, order is lat then lng
                            lat = float(matches.group(1))
                            lng = float(matches.group(2))
                        
                        warning_location = (lat, lng)
                        logger.info(f"Successfully extracted coordinates from HTML: {lat}, {lng}")
                        break
            except Exception as e:
                logger.error(f"Error with direct HTML extraction: {e}")
        
        if not warning_location:
            logger.info(f"Could not extract location from warning, skipping broadcast")
            return {'broadcast': False, 'reason': 'invalid_location'}
            
        warning_lat, warning_lng = warning_location
        logger.info(f"Warning location: {warning_lat}, {warning_lng}")
        
        # Get all user notification preferences
        settings_ref = db.collection('settings').where('broadcast.enabled', '==', True)
        
        # Track number of notifications sent
        broadcast_count = 0
        user_count = 0
        
        # Store original warning owner ID
        original_owner_uid = warning.get('owner_uid')
        logger.info(f"Original warning owner: {original_owner_uid}")
        
        for doc in settings_ref.stream():
            try:
                user_prefs = doc.to_dict()
                user_id = user_prefs.get('owner_uid')
                
                if not user_id:
                    logger.info(f"Skipping document without owner_uid: {doc.id}")
                    continue
                
                # Skip if this is the user who triggered the warning
                if user_id == original_owner_uid:
                    logger.info(f"Skipping broadcast to original warning owner: {user_id}")
                    continue
                    
                broadcast_settings = user_prefs.get('broadcast', {})
                user_location_str = broadcast_settings.get('location', '')
                
                # Skip if user has no location set
                if not user_location_str:
                    logger.info(f"User {user_id} has no location set, skipping")
                    continue
                    
                # Parse user location using improved method
                user_location = LocationUtils.extract_coordinates(user_location_str)
                if not user_location:
                    logger.info(f"Could not parse location for user {user_id}, skipping")
                    continue
                    
                user_lat, user_lng = user_location
                logger.info(f"User {user_id} location: {user_lat}, {user_lng}")
                
                # Calculate distance using improved method
                distance = LocationUtils.calculate_distance(
                    (warning_lat, warning_lng), 
                    (user_lat, user_lng)
                )
                
                if distance is None:
                    logger.warning(f"Failed to calculate distance for user {user_id}")
                    continue
                
                # Format distance for display (without units)
                distance_str = LocationUtils.format_distance_for_display(distance, with_units=False)
                
                logger.info(f"User {user_id} is {distance_str}km from warning location")
                
                # Check if within radius - handle both string and int types
                try:
                    user_radius = broadcast_settings.get('radius', 10)
                    # Convert to float if it's a string
                    if isinstance(user_radius, str):
                        user_radius = float(user_radius)
                    
                    # Ensure we have a valid number
                    if not isinstance(user_radius, (int, float)) or user_radius <= 0:
                        user_radius = 10  # Default to 10km
                        
                    logger.info(f"User {user_id} has radius set to {user_radius}km")
                    
                    if distance > user_radius:
                        logger.info(f"User {user_id} is outside broadcast radius ({distance}km > {user_radius}km), skipping")
                        continue
                        
                except (ValueError, TypeError) as e:
                    logger.error(f"Error parsing radius for user {user_id}: {e}")
                    # Default to 10km if we can't parse the radius
                    user_radius = 10
                    if distance > user_radius:
                        logger.info(f"User {user_id} is outside default radius, skipping")
                        continue
                    
                # Check if user is interested in this animal type
                animal_type = warning.get('type', '').lower()
                user_animals = broadcast_settings.get('animals', ['all'])
                
                logger.info(f"User {user_id} is interested in animals: {user_animals}")
                logger.info(f"Current animal type: {animal_type}")
                
                if 'all' not in user_animals and animal_type not in user_animals:
                    logger.info(f"User {user_id} is not interested in {animal_type}, skipping")
                    continue
                    
                logger.info(f"Preparing broadcast for user {user_id} - within radius and interested in {animal_type}")
                
                # User is within radius and interested in this animal type
                # Prepare broadcast notification
                broadcast_methods = broadcast_settings.get('methods', [])
                
                # Create a modified message for broadcast
                broadcast_message = f"🚨 NEARBY ALERT: {warning['type']} detected {distance_str}km from your location\n\n"
                broadcast_message += f"Severity: {warning['severity'].upper()}\n"
                broadcast_message += f"Time: {warning.get('formatted_date', warning.get('formatted_timestamp', 'Unknown'))}\n"
                broadcast_message += f"Detected by: Another WildEye user\n"
                # Format distance with proper units for the message
                broadcast_message += f"Distance: {LocationUtils.format_distance_for_display(distance, with_units=True)} from your location\n"
                
                # Create a copy of warning with modified message for broadcast
                broadcast_warning_copy = warning.copy()
                broadcast_warning_copy['message'] = broadcast_message
                broadcast_warning_copy['is_broadcast'] = True  # Explicitly set this flag
                broadcast_warning_copy['distance'] = distance_str
                
                # Add a new field in database for the warning document 
                broadcast_ref = db.collection('warnings').document()
                broadcast_id = broadcast_ref.id
                
                # Create broadcast warning record with all necessary fields
                broadcast_record = {
                    'warning_id': broadcast_id,
                    'detection_id': broadcast_warning_copy.get('detection_id', ''),
                    'camera_id': broadcast_warning_copy.get('camera_id', ''),
                    'camera_name': broadcast_warning_copy.get('camera_name', ''),
                    'type': broadcast_warning_copy.get('type', ''),
                    'detection_label': broadcast_warning_copy.get('detection_label', ''),
                    'message': broadcast_message,
                    'screenshot_url': broadcast_warning_copy.get('screenshot_url', ''),
                    'google_maps_link': broadcast_warning_copy.get('google_maps_link', ''),
                    'severity': broadcast_warning_copy.get('severity', 'medium'),
                    'timestamp': datetime.now(),
                    'formatted_date': format_date_dmy(datetime.now()),
                    'active': True,
                    'acknowledged': False,
                    'is_broadcast': True,  # This is a broadcast alert
                    'distance': distance_str,  # Store numeric value without units
                    'distance_km': distance,   # Store the actual numeric distance value
                    'original_warning_id': warning.get('warning_id', ''),  # Reference to original
                    'owner_uid': user_id,  # This should be the recipient's ID
                    'notification_status': {
                        'email': False,
                        'sms': False,
                        'telegram': False,
                        'call': False
                    }
                }
                
                # Save broadcast warning to database
                broadcast_ref.set(broadcast_record)
                logger.info(f"Created broadcast warning {broadcast_id} for user {user_id}")
                
                # Create special notification preferences based on user's broadcast methods
                broadcast_prefs = {
                    'email': {
                        'enabled': 'email' in broadcast_methods,
                        'recipient': user_prefs.get('email', {}).get('recipient', '')
                    },
                    'sms': {
                        'enabled': 'sms' in broadcast_methods,
                        'recipient': user_prefs.get('sms', {}).get('recipient', ''),
                        'country_code': user_prefs.get('sms', {}).get('country_code', '+91')
                    },
                    'telegram': {
                        'enabled': 'telegram' in broadcast_methods,
                        'chat_id': user_prefs.get('telegram', {}).get('chat_id', '')
                    },
                    'call': {
                        'enabled': 'call' in broadcast_methods and warning.get('severity') == 'high',  # Only call for high severity
                        'recipient': user_prefs.get('call', {}).get('recipient', ''),
                        'country_code': user_prefs.get('call', {}).get('country_code', '+91')
                    }
                }
                
                # Log the broadcast preferences
                logger.info(f"Broadcast preferences for user {user_id}: {broadcast_prefs}")
                
                # Send notifications
                notification_status = send_notifications(broadcast_record, broadcast_prefs)
                
                # Log the notification status
                logger.info(f"Broadcast notification status for user {user_id}: {notification_status}")
                
                # Update notification status in the database
                broadcast_ref.update({
                    'notification_status': notification_status
                })
                
                # Count broadcasts
                if any(notification_status.values()):
                    broadcast_count += sum(1 for status in notification_status.values() if status)
                    user_count += 1
                    
                    # Log the broadcast
                    logger.info(f"Broadcast alert sent to user {user_id} ({distance_str}km away)")
                
            except Exception as user_error:
                logger.error(f"Error processing broadcast for user: {user_error}")
                continue
                
        logger.info(f"Broadcast complete. Notified {user_count} users with {broadcast_count} notifications")
        return {
            'broadcast': True,
            'users_notified': user_count,
            'notifications_sent': broadcast_count
        }
        
    except Exception as e:
        logger.error(f"Error in broadcast_warning: {e}")
        return {'broadcast': False, 'error': str(e)}

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
        
        # Add debug logging for broadcast alerts
        is_broadcast = warning and warning.get('is_broadcast', False)
        logger.info(f"Sending Telegram notification to chat_id: {chat_id}, is_broadcast: {is_broadcast}")
        
        # If we have a full warning object, use the enhanced notification function
        if warning:
            # Make sure screenshot_url is included in the warning data
            if screenshot_url and 'screenshot_url' not in warning:
                warning['screenshot_url'] = screenshot_url
                
            # Call the enhanced function from telegram_service
            result = send_telegram_notification(chat_id, warning)
            logger.info(f"Telegram notification result: {result}")
            return result
        
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
            
            result = send_telegram_notification(chat_id, minimal_warning)
            logger.info(f"Telegram notification result: {result}")
            return result
            
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
            },
            'broadcast': {
                'enabled': False,
                'location': '',
                'radius': 10,           # Default radius in km
                'animals': ['all'],     # Default to all animals
                'methods': ['email', 'sms']  # Default notification methods
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
            
            # Ensure broadcast settings exist for older records
            if 'broadcast' not in preferences:
                preferences['broadcast'] = default_preferences['broadcast']
                
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