import logging
import os
import requests
import re
import time
from typing import Dict, Optional, Union
from datetime import datetime
import io

# Configure logging
logger = logging.getLogger(__name__)

# Telegram API base URL
TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/{method}"

# Get Telegram bot token from environment
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
EMERGENCY_CONTACT = os.environ.get('EMERGENCY_CONTACT', 'Forest Dept: XXX-XXX-XXXX')

# Animal-specific templates for Telegram messages
ANIMAL_TELEGRAM_TEMPLATES = {
    'default': """
🚨 *WildEye Alert*: Animal Detected 🚨

An unidentified animal has been detected at {camera_name}.

*General precautions:*
• Do not approach the animal
• Keep a safe distance
• Contact local authorities if needed

Emergency Contact: {emergency_contact}
""",
    'tiger': """
🚨 *URGENT DANGER*: Tiger Detected 🚨

A tiger has been detected at {camera_name}.

*IMMEDIATE ACTIONS REQUIRED:*
• Stay indoors and secure all doors and windows
• Keep all pets and livestock secured
• Do NOT approach under any circumstances
• Contact forest department immediately: {emergency_contact}
• Alert neighbors about the tiger's presence

⚠️ Tigers are apex predators and extremely dangerous
""",
    'leopard': """
🚨 *URGENT DANGER*: Leopard Detected 🚨

A leopard has been detected at {camera_name}.

*IMMEDIATE ACTIONS REQUIRED:*
• Stay indoors and secure all doors and windows
• Keep all pets and livestock secured
• Do NOT approach under any circumstances
• Contact forest department immediately: {emergency_contact}

⚠️ Leopards can climb trees and buildings - be extra vigilant
""",
    'elephant': """
🚨 *WARNING*: Elephant Detected 🚨

An elephant has been detected at {camera_name}.

*Take these precautions:*
• Stay indoors and remain quiet
• Avoid sudden movements or loud noises
• Do NOT approach - elephants require significant space
• Contact forest department: {emergency_contact}

⚠️ Elephants can cause significant property damage
""",
    'bear': """
🚨 *DANGER*: Bear Detected 🚨

A bear has been detected at {camera_name}.

*IMMEDIATE ACTIONS REQUIRED:*
• Stay indoors and secure all doors and windows
• Remove or secure all food sources outside
• Do NOT approach the animal
• Contact forest department immediately: {emergency_contact}

⚠️ Bears are particularly dangerous if they feel threatened
""",
    'wild boar': """
🚨 *CAUTION*: Wild Boar Detected 🚨

A wild boar has been detected at {camera_name}.

*Take these precautions:*
• Stay indoors if the animal is near
• Keep pets and children away from the area
• Wild boars may charge if threatened
• Contact wildlife authorities if aggressive: {emergency_contact}
""",
    'wild buffalo': """
🚨 *DANGER*: Wild Buffalo Detected 🚨

A wild buffalo has been detected at {camera_name}.

*IMMEDIATE ACTIONS REQUIRED:*
• Stay indoors and secure all doors
• Keep far away - wild buffaloes can be aggressive
• Do NOT approach under any circumstances
• Contact forest department immediately: {emergency_contact}

⚠️ Wild buffaloes are unpredictable and dangerous
""",
    'lion': """
🚨 *EXTREME DANGER*: Lion Detected 🚨

A lion has been detected at {camera_name}.

*IMMEDIATE ACTIONS REQUIRED:*
• Stay indoors and secure all entrances immediately
• Keep all pets and livestock secured
• Do NOT approach under any circumstances
• Contact emergency services immediately: {emergency_contact}
• Alert all neighbors about the lion's presence

⚠️ This is an extremely dangerous situation
"""
}

# Broadcast alert templates
BROADCAST_TELEGRAM_TEMPLATES = {
    'default': """
🔔 *NEARBY ALERT*: Animal Detected {distance}km Away 🔔

An unidentified animal has been detected {distance}km from your location.

*General precautions:*
• Be aware of your surroundings
• Keep a safe distance if venturing outside
• Contact local authorities if the animal approaches

This alert was generated from another WildEye user's camera.
""",
    'tiger': """
🚨 *URGENT DANGER*: Tiger Detected {distance}km Away 🚨

A tiger has been detected {distance}km from your location.

*IMMEDIATE ACTIONS RECOMMENDED:*
• Stay alert and be cautious when outside
• Keep all pets and livestock secured
• Do NOT approach if sighted
• Contact forest department immediately if sighted: {emergency_contact}

⚠️ Tigers are apex predators and extremely dangerous
This alert was generated from another WildEye user's camera.
""",
    'leopard': """
🚨 *URGENT DANGER*: Leopard Detected {distance}km Away 🚨

A leopard has been detected {distance}km from your location.

*IMMEDIATE ACTIONS RECOMMENDED:*
• Stay alert and be cautious when outside
• Keep all pets and livestock secured
• Do NOT approach if sighted
• Contact forest department immediately if sighted: {emergency_contact}

⚠️ Leopards can climb trees and buildings - be extra vigilant
This alert was generated from another WildEye user's camera.
""",
    'elephant': """
🚨 *WARNING*: Elephant Detected {distance}km Away 🚨

An elephant has been detected {distance}km from your location.

*Take these precautions:*
• Be cautious when traveling in the area
• Keep a significant distance if sighted
• Do NOT approach - elephants require significant space
• Contact forest department if sighted: {emergency_contact}

⚠️ Elephants can cause significant property damage
This alert was generated from another WildEye user's camera.
""",
    'bear': """
🚨 *DANGER*: Bear Detected {distance}km Away 🚨

A bear has been detected {distance}km from your location.

*ACTIONS RECOMMENDED:*
• Be cautious when traveling in the area
• Secure food sources if camping or outdoors
• Do NOT approach if sighted
• Contact forest department immediately if sighted: {emergency_contact}

⚠️ Bears are particularly dangerous if they feel threatened
This alert was generated from another WildEye user's camera.
""",
    'wild boar': """
🚨 *CAUTION*: Wild Boar Detected {distance}km Away 🚨

A wild boar has been detected {distance}km from your location.

*Take these precautions:*
• Be cautious when traveling in the area
• Keep pets supervised when outdoors
• Wild boars may charge if threatened
• Contact wildlife authorities if aggressive: {emergency_contact}

This alert was generated from another WildEye user's camera.
""",
    'wild buffalo': """
🚨 *DANGER*: Wild Buffalo Detected {distance}km Away 🚨

A wild buffalo has been detected {distance}km from your location.

*ACTIONS RECOMMENDED:*
• Be extremely cautious when traveling in the area
• Keep far away if sighted - wild buffaloes can be aggressive
• Do NOT approach under any circumstances
• Contact forest department immediately if sighted: {emergency_contact}

⚠️ Wild buffaloes are unpredictable and dangerous
This alert was generated from another WildEye user's camera.
""",
    'lion': """
🚨 *EXTREME DANGER*: Lion Detected {distance}km Away 🚨

A lion has been detected {distance}km from your location.

*ACTIONS RECOMMENDED:*
• Avoid traveling in the area if possible
• Travel only in vehicles with closed windows
• Do NOT approach under any circumstances
• Contact emergency services immediately if sighted: {emergency_contact}

⚠️ This is an extremely dangerous situation
This alert was generated from another WildEye user's camera.
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

def get_animal_telegram_template(animal_type: str, is_broadcast: bool = False, distance: str = None) -> str:
    """
    Get the appropriate Telegram template for the specified animal type.
    
    Args:
        animal_type: The type of animal detected
        is_broadcast: Whether this is a broadcast alert from another user's camera
        distance: Distance from user's location (for broadcast alerts)
        
    Returns:
        str: Telegram message template for the animal
    """
    animal_type = animal_type.lower()
    
    # Use broadcast templates for broadcast alerts
    if is_broadcast:
        templates = BROADCAST_TELEGRAM_TEMPLATES
        # Default distance if not provided
        distance = distance or "unknown"
        
        # Check for specific animal matches
        for animal in templates:
            if animal in animal_type:
                return templates[animal].format(distance=distance, emergency_contact=EMERGENCY_CONTACT)
        
        # Return default template if no match
        return templates['default'].format(distance=distance, emergency_contact=EMERGENCY_CONTACT)
    else:
        # Use regular templates for direct alerts
        templates = ANIMAL_TELEGRAM_TEMPLATES
        
        # Check for specific animal matches
        for animal in templates:
            if animal in animal_type:
                return templates[animal]
        
        # Return default template if no match
        return templates['default']

def convert_google_drive_url(url: str) -> str:
    """
    Convert a Google Drive sharing URL to a direct download URL.
    
    Args:
        url: Google Drive sharing URL
        
    Returns:
        str: Direct download URL or original URL if not a Google Drive URL
    """
    try:
        if 'drive.google.com/file/d/' in url:
            # Extract the file ID from the URL
            # Example URL: https://drive.google.com/file/d/1oG30PKz6OGWS36XI6JPCPm77-tkYLkkK/view?usp=drivesdk
            file_id = url.split('/file/d/')[1].split('/')[0]
            
            # Create direct download URL
            direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            logger.info(f"Converted Google Drive URL to direct URL. ID: {file_id}")
            return direct_url
        
        return url
    except Exception as e:
        logger.error(f"Failed to convert Google Drive URL: {e}")
        return url

def download_image(url, save_path):
    """Download an image from a URL
    
    Args:
        url: URL of the image to download
        save_path: Path where to save the image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert Google Drive URL if needed
        direct_url = convert_google_drive_url(url)
        
        logger.info(f"Downloading image from {direct_url}")
        
        # Set headers to mimic a browser request (sometimes needed for Google Drive)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Use a session to handle cookies and redirects properly
        with requests.Session() as session:
            response = session.get(direct_url, stream=True, timeout=30, headers=headers)
            
            # If we received a small file, it might be the Google warning page
            # instead of the actual image
            if 'drive.google.com' in direct_url and len(response.content) < 10000:
                # Try the alternative export URL format
                file_id = direct_url.split('id=')[1]
                alt_url = f"https://drive.google.com/uc?id={file_id}&export=download"
                logger.info(f"First attempt failed, trying alternative Google Drive URL: {alt_url}")
                response = session.get(alt_url, stream=True, timeout=30, headers=headers)
        
            response.raise_for_status()
            
            # Check if the content type is an image
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                logger.warning(f"Downloaded content is not an image (Content-Type: {content_type})")
                
                # For Google Drive, sometimes we need to confirm a download
                if 'drive.google.com' in direct_url and 'text/html' in content_type:
                    logger.info("Received HTML from Google Drive, attempting to extract direct image URL")
                    # This is a complex case requiring additional handling
                    # For now, we'll save the content and proceed, but this may not work
            
            # Save the file
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    
            logger.info(f"Image downloaded to {save_path}")
            
            # Verify the downloaded file is valid
            if os.path.getsize(save_path) < 100:  # Very small file, likely an error
                logger.error(f"Downloaded file is too small ({os.path.getsize(save_path)} bytes)")
                return False
                
            return True
        
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        return False

def send_telegram_photo(chat_id: str, photo_url: str, caption: str = None) -> bool:
    """
    Send a photo to a Telegram chat.
    
    Args:
        chat_id: Telegram chat ID
        photo_url: URL of the photo to send
        caption: Optional caption for the photo
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not TELEGRAM_BOT_TOKEN:
            logger.warning("Missing Telegram bot token")
            return False
        
        if not photo_url:
            logger.error("Empty photo URL provided")
            return False
            
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        
        # Log the attempt with partial URL for debugging
        logger.info(f"Sending photo with URL: {photo_url[:30]}... to chat {chat_id}")
        
        # Prepare payload
        data = {
            'chat_id': chat_id,
            'photo': photo_url,
            'parse_mode': 'Markdown'
        }
        
        if caption:
            # Limit caption length to Telegram's maximum (1024 characters)
            if len(caption) > 1024:
                logger.warning(f"Caption too long ({len(caption)} chars), truncating to 1024 chars")
                caption = caption[:1020] + "..."
            data['caption'] = caption
        
        # Add timeout to prevent hanging
        timeout = 30  # seconds
            
        # Send request
        response = requests.post(url, data=data, timeout=timeout)
        
        # Log the response status
        logger.info(f"Response status code: {response.status_code}")
        
        # Check response
        if response.status_code != 200:
            logger.error(f"Telegram API error: {response.status_code} - {response.text}")
            return False
            
        response_data = response.json()
        if not response_data.get('ok', False):
            logger.error(f"Telegram API returned error: {response_data.get('description', 'Unknown error')}")
            return False
        
        # If successful, get the message_id from the response
        message_id = response_data.get('result', {}).get('message_id')
        logger.info(f"Successfully sent photo via Telegram (message_id: {message_id})")
        return True
        
    except requests.exceptions.Timeout:
        logger.error("Telegram API request timed out when sending photo")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error when sending Telegram photo: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to send Telegram photo: {e}")
        return False

def send_photo_by_file(chat_id: str, file_path: str, caption: str = None) -> bool:
    """Send a photo to Telegram by uploading a file
    
    Args:
        chat_id: Telegram chat ID
        file_path: Path to the image file to send
        caption: Optional caption for the photo
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    data = {
        'chat_id': chat_id,
        'parse_mode': 'Markdown'
    }
    
    if caption:
        # Limit caption length to Telegram's maximum (1024 characters)
        if len(caption) > 1024:
            logger.warning(f"Caption too long ({len(caption)} chars), truncating to 1024 chars")
            caption = caption[:1020] + "..."
        data['caption'] = caption
    
    try:
        logger.info(f"Sending photo file to chat ID: {chat_id}")
        logger.info(f"File path: {file_path}")
        
        with open(file_path, 'rb') as photo_file:
            files = {'photo': photo_file}
            response = requests.post(url, data=data, files=files, timeout=30)
        
        # Log the response status
        logger.info(f"Response status code: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Failed to send photo by file: {response.status_code} - {response.text}")
            return False
        else:
            logger.info("Photo sent successfully by file upload")
            return True
            
    except Exception as e:
        logger.error(f"Error sending photo by file: {e}")
        return False

def send_telegram_message(chat_id: str, message: str) -> bool:
    """
    Send a text message to a Telegram chat.
    
    Args:
        chat_id: Telegram chat ID
        message: Message text
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not TELEGRAM_BOT_TOKEN:
            logger.warning("Missing Telegram bot token")
            return False
            
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        
        # Limit message length to Telegram's maximum (4096 characters)
        if len(message) > 4096:
            message = message[:4093] + "..."
            
        # Prepare payload
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        # Add timeout to prevent hanging
        timeout = 30  # seconds
            
        # Send request
        response = requests.post(url, data=payload, timeout=timeout)
        
        # Check response
        if response.status_code != 200:
            logger.error(f"Telegram API error: {response.status_code} - {response.text}")
            return False
            
        response_data = response.json()
        if not response_data.get('ok', False):
            logger.error(f"Telegram API returned error: {response_data.get('description', 'Unknown error')}")
            return False
        
        logger.info(f"Successfully sent message via Telegram: {message[:50]}...")
        return True
        
    except requests.exceptions.Timeout:
        logger.error("Telegram API request timed out when sending message")
        return False
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False

def send_telegram_notification(chat_id: str, detection_data: Dict) -> bool:
    """
    Send a comprehensive Telegram notification for an animal detection.
    
    Args:
        chat_id: Telegram chat ID
        detection_data: Dictionary with detection details
        
    Returns:
        bool: True if notification was sent successfully
    """
    try:
        if not TELEGRAM_BOT_TOKEN or not chat_id:
            logger.warning("Missing Telegram credentials or chat ID")
            return False
            
        # Check if this is a broadcast alert
        is_broadcast = detection_data.get('is_broadcast', False)
        distance = detection_data.get('distance', "unknown") if is_broadcast else None
            
        # Get animal details
        animal_type = detection_data.get('detection_label', '')
        
        # If empty, try 'type' (from warning_system.py)
        if not animal_type:
            animal_type = detection_data.get('type', '')
            
        # If still empty, use default
        if not animal_type:
            animal_type = 'unknown'
            
        # Get camera name
        camera_name = detection_data.get('camera_name', 'Unknown location')
            
        # Format timestamp - separate date and time
        date_str = ""
        time_str = ""
        timestamp = detection_data.get('timestamp', datetime.now())
        
        # Format the date and time separately
        if isinstance(timestamp, datetime):
            date_str = timestamp.strftime("%d-%m-%Y")
            time_str = timestamp.strftime("%I:%M:%S %p")
        else:
            # Try to parse from formatted_date or formatted_timestamp
            if 'formatted_date' in detection_data:
                parts = detection_data['formatted_date'].split(' ')
                if len(parts) >= 2:
                    date_str = parts[0]
                    time_str = ' '.join(parts[1:])
            elif 'formatted_timestamp' in detection_data:
                parts = detection_data['formatted_timestamp'].split(' ')
                if len(parts) >= 2:
                    date_str = parts[0]
                    time_str = ' '.join(parts[1:])
            else:
                date_str = "Unknown date"
                time_str = "Unknown time"
            
        # Get location info
        location_link = detection_data.get('google_maps_link', '')
        
        # Get screenshot URL
        screenshot_url = detection_data.get('screenshot_url', '')
        
        # Get animal-specific template
        if is_broadcast:
            template = get_animal_telegram_template(animal_type, is_broadcast=True, distance=distance)
        else:
            template = get_animal_telegram_template(animal_type)
        
        # Construct the message
        if is_broadcast:
            message = template
        else:
            message = template.format(
                camera_name=camera_name,
                emergency_contact=EMERGENCY_CONTACT
            )
        
        # Add date and time information separately
        message += f"\n*Date:* {date_str}"
        message += f"\n*Time:* {time_str}"
        
        if location_link:
            message += f"\n*Location:* [View on map]({location_link})"
            
        # Add confidence if available
        if 'confidence' in detection_data:
            confidence = detection_data['confidence']
            if isinstance(confidence, (int, float)):
                message += f"\n*Confidence:* {confidence:.1f}%"
        
        # Send notification with photo if available
        notification_sent = False
        
        if screenshot_url:
            # Check if it's a Google Drive URL and convert if needed
            is_google_drive = 'drive.google.com' in screenshot_url
            
            # Try multiple methods to send the image with caption
            photo_methods = []
            
            # Method 1: Google Drive thumbnail (if applicable)
            if is_google_drive:
                try:
                    file_id = screenshot_url.split('/file/d/')[1].split('/')[0]
                    thumbnail_url = f"https://drive.google.com/thumbnail?id={file_id}&sz=w1000"
                    logger.info(f"Using Google Drive thumbnail URL: {thumbnail_url}")
                    photo_methods.append(thumbnail_url)
                except Exception as e:
                    logger.error(f"Failed to create Google Drive thumbnail URL: {e}")
            
            # Method 2: Direct URL or converted URL
            direct_url = convert_google_drive_url(screenshot_url)
            if direct_url != screenshot_url:  # Only add if it's different from the original
                photo_methods.append(direct_url)
            
            # Method 3: Original URL
            photo_methods.append(screenshot_url)
            
            # Method 4: Google Drive open URL (if applicable)
            if is_google_drive:
                try:
                    file_id = screenshot_url.split('/file/d/')[1].split('/')[0]
                    open_url = f"https://drive.google.com/open?id={file_id}"
                    photo_methods.append(open_url)
                except Exception as e:
                    logger.error(f"Failed to create Google Drive open URL: {e}")
            
            # Try each URL method one by one
            for photo_url in photo_methods:
                logger.info(f"Attempting to send notification with photo URL: {photo_url[:50]}...")
                
                # Try to send the photo with the full message as caption
                notification_sent = send_telegram_photo(chat_id, photo_url, message)
                
                if notification_sent:
                    logger.info("Successfully sent notification with photo")
                    break
            
            # If all URL methods failed, try the download and file upload method
            if not notification_sent:
                logger.warning("All URL methods failed. Trying to download and send as file.")
                
                # Create a temp file name with timestamp to avoid conflicts
                temp_file = f"temp_screenshot_{int(time.time())}.jpg"
                
                # Try each URL for downloading
                for photo_url in photo_methods:
                    # Download the image
                    download_success = download_image(photo_url, temp_file)
                    
                    if download_success and os.path.exists(temp_file):
                        # Try to send the photo as a file with full message as caption
                        notification_sent = send_photo_by_file(chat_id, temp_file, message)
                        
                        if notification_sent:
                            logger.info("Successfully sent notification with photo as file")
                            break
                
                # Clean up the temp file
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                        logger.info(f"Temporary file removed: {temp_file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file: {e}")
        
        # If photo methods failed or no photo was available, send text-only message
        if not notification_sent:
            if screenshot_url:
                logger.error(f"Failed to send notification with photo for {animal_type} detection")
                # Add a note about the missing image to the message
                message += "\n\n⚠️ *Note:* System could not deliver the detection image."
            else:
                logger.warning(f"No screenshot URL provided for {animal_type} detection")
                
            # Send as text-only message
            notification_sent = send_telegram_message(chat_id, message)
        
        if notification_sent:
            logger.info(f"Telegram notification sent successfully for {animal_type} detection")
            return True
        else:
            logger.error(f"Failed to send any Telegram notification for {animal_type} detection")
            return False
        
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def send_telegram_test(chat_id: str) -> bool:
    """
    Send a test message to verify Telegram configuration.
    
    Args:
        chat_id: Telegram chat ID to test
        
    Returns:
        bool: True if test message was sent successfully
    """
    try:
        timestamp = format_date_dmy(datetime.now())
        
        message = f"""
🧪 *WildEye Test Notification*

This is a test notification from your WildEye wildlife detection system.
If you're receiving this message, your Telegram notifications are correctly configured.

*System Time:* {timestamp}

To manage notification settings, please access your WildEye dashboard.
"""
        
        # Create a combined test message with both text and image
        combined_sent = False
        
        # Try to send a test image with the message as caption
        test_image_url = "https://via.placeholder.com/800x600.png?text=WildEye+Test"
        combined_sent = send_telegram_photo(chat_id, test_image_url, message)
        
        # If combined message fails, fall back to separate messages
        if not combined_sent:
            logger.warning("Failed to send combined test message, trying text-only")
            message_sent = send_telegram_message(chat_id, message)
            
            if message_sent:
                logger.info(f"Telegram test notification (text-only) sent successfully to chat {chat_id}")
                return True
            else:
                logger.error("Failed to send any Telegram test messages")
                return False
        else:
            logger.info(f"Combined Telegram test notification sent successfully to chat {chat_id}")
            return True
        
    except Exception as e:
        logger.error(f"Failed to send Telegram test: {e}")
        return False