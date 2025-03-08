# telegram_service.py
import logging
import os
import requests
from datetime import datetime
from typing import Dict, Optional, List
import io

# Configure logging
logger = logging.getLogger(__name__)

# Telegram credentials (load from environment variables)
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

def get_animal_telegram_template(animal_type: str) -> str:
    """
    Get the appropriate Telegram template for the specified animal type.
    
    Args:
        animal_type: The type of animal detected
        
    Returns:
        str: Telegram message template for the animal
    """
    animal_type = animal_type.lower()
    
    # Check for specific animal matches
    for animal in ANIMAL_TELEGRAM_TEMPLATES:
        if animal in animal_type:
            return ANIMAL_TELEGRAM_TEMPLATES[animal]
    
    # Return default template if no match
    return ANIMAL_TELEGRAM_TEMPLATES['default']

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
            
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        
        # Prepare payload
        payload = {
            'chat_id': chat_id,
            'photo': photo_url,
            'parse_mode': 'Markdown'
        }
        
        if caption:
            # Limit caption length to Telegram's maximum (1024 characters)
            if len(caption) > 1024:
                caption = caption[:1021] + "..."
            payload['caption'] = caption
            
        # Send request
        response = requests.post(url, data=payload)
        response.raise_for_status()
        
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram photo: {e}")
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
            
        # Send request
        response = requests.post(url, data=payload)
        response.raise_for_status()
        
        return True
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
            
        # Get location info
        location_link = detection_data.get('google_maps_link', '')
        
        # Get screenshot URL
        screenshot_url = detection_data.get('screenshot_url', '')
        
        # Get animal-specific template
        template = get_animal_telegram_template(animal_type)
        
        # Construct the message
        message = template.format(
            camera_name=camera_name,
            emergency_contact=EMERGENCY_CONTACT
        )
        
        # Add time and location information
        message += f"\n*Time:* {timestamp_str}"
        
        if location_link:
            message += f"\n*Location:* [View on map]({location_link})"
            
        # Add confidence if available
        if 'confidence' in detection_data:
            confidence = detection_data['confidence']
            if isinstance(confidence, (int, float)):
                message += f"\n*Confidence:* {confidence:.1f}%"
        
        # Send message with photo if available
        if screenshot_url:
            # Send photo with caption
            photo_caption = f"🚨 {animal_type.title()} detected at {camera_name}"
            send_telegram_photo(chat_id, screenshot_url, photo_caption)
            
            # Send detailed message separately
            result = send_telegram_message(chat_id, message)
        else:
            # Send text message only
            result = send_telegram_message(chat_id, message)
            
        if result:
            logger.info(f"Telegram notification sent to chat {chat_id} for {animal_type} detection")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")
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
        
        result = send_telegram_message(chat_id, message)
        
        if result:
            logger.info(f"Telegram test notification sent successfully to chat {chat_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to send Telegram test: {e}")
        return False