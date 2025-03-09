# email_service.py
import logging
import smtplib
import os
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import requests
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Union, BinaryIO

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables or config
EMAIL_SERVER = os.environ.get('EMAIL_SERVER', 'smtp.gmail.com')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', 587))
EMAIL_USERNAME = os.environ.get('EMAIL_USERNAME', '')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', '')
EMERGENCY_CONTACT = os.environ.get('EMERGENCY_CONTACT', 'Forest Dept: XXX-XXX-XXXX')

# Animal details and precautions for email
ANIMAL_EMAIL_TEMPLATES = {
    'default': """
        <h3>Animal Detection Alert</h3>
        <p>An animal has been detected on your property. Please exercise caution.</p>
        <p>General precautions:</p>
        <ul>
            <li>Do not approach the animal</li>
            <li>Keep a safe distance</li>
            <li>Contact local wildlife authorities if needed</li>
        </ul>
    """,
    'tiger': """
        <h3>DANGER: Tiger Detection Alert</h3>
        <p>A tiger has been detected near your camera. This is a potentially dangerous situation.</p>
        <p>Take these immediate actions:</p>
        <ul>
            <li>Stay indoors and secure all doors and windows</li>
            <li>Keep all pets and livestock secured in enclosed spaces</li>
            <li>Do NOT approach or attempt to photograph the animal</li>
            <li>Contact forest department immediately at: {emergency_contact}</li>
            <li>Alert neighbors about the presence of the tiger</li>
        </ul>
        <p>Tigers are apex predators and should be treated with extreme caution.</p>
    """,
    'leopard': """
        <h3>DANGER: Leopard Detection Alert</h3>
        <p>A leopard has been detected near your camera. This is a potentially dangerous situation.</p>
        <p>Take these immediate actions:</p>
        <ul>
            <li>Stay indoors and secure all doors and windows</li>
            <li>Keep all pets and livestock secured in enclosed spaces</li>
            <li>Do NOT approach or attempt to photograph the animal</li>
            <li>Contact forest department immediately at: {emergency_contact}</li>
            <li>Leopards can climb trees and buildings - be extra vigilant</li>
        </ul>
    """,
    'elephant': """
        <h3>WARNING: Elephant Detection Alert</h3>
        <p>An elephant has been detected near your camera.</p>
        <p>Take these precautions:</p>
        <ul>
            <li>Stay indoors and remain quiet</li>
            <li>Avoid sudden movements or loud noises that might startle the elephant</li>
            <li>Do NOT approach the elephant - they require significant space</li>
            <li>Contact forest department at: {emergency_contact}</li>
            <li>Elephants can cause significant property damage - secure loose items</li>
        </ul>
    """,
    'bear': """
        <h3>DANGER: Bear Detection Alert</h3>
        <p>A bear has been detected near your camera. This is a potentially dangerous situation.</p>
        <p>Take these immediate actions:</p>
        <ul>
            <li>Stay indoors and secure all doors and windows</li>
            <li>Remove or secure all food sources outside (trash, pet food, bird feeders)</li>
            <li>Do NOT approach or attempt to photograph the animal</li>
            <li>Contact forest department immediately at: {emergency_contact}</li>
            <li>Bears are particularly dangerous if they feel threatened or have cubs</li>
        </ul>
    """,
    'wild boar': """
        <h3>CAUTION: Wild Boar Detection Alert</h3>
        <p>A wild boar has been detected near your camera.</p>
        <p>Take these precautions:</p>
        <ul>
            <li>Stay indoors if the animal is near your dwelling</li>
            <li>Keep pets and children away from the area</li>
            <li>Wild boars can cause property damage and may charge if threatened</li>
            <li>Contact wildlife authorities if the animal appears aggressive: {emergency_contact}</li>
        </ul>
    """,
    'wild buffalo': """
        <h3>DANGER: Wild Buffalo Detection Alert</h3>
        <p>A wild buffalo has been detected near your camera. This is a potentially dangerous situation.</p>
        <p>Take these immediate actions:</p>
        <ul>
            <li>Stay indoors and secure all doors and windows</li>
            <li>Keep far away - wild buffaloes can be aggressive and unpredictable</li>
            <li>Do NOT approach the animal under any circumstances</li>
            <li>Alert others in the area about the presence of the buffalo</li>
            <li>Contact forest department immediately at: {emergency_contact}</li>
        </ul>
    """,
    'lion': """
        <h3>EXTREME DANGER: Lion Detection Alert</h3>
        <p>A lion has been detected near your camera. This is an extremely dangerous situation.</p>
        <p>Take these immediate actions:</p>
        <ul>
            <li>Stay indoors and secure all doors and windows immediately</li>
            <li>Keep all pets and livestock secured in enclosed spaces</li>
            <li>Do NOT approach or attempt to photograph the animal under any circumstances</li>
            <li>Contact forest department/emergency services immediately at: {emergency_contact}</li>
            <li>Alert all neighbors about the presence of the lion</li>
            <li>If you must move outdoors, travel in a vehicle with closed windows</li>
        </ul>
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

def get_animal_email_template(animal_type: str) -> str:
    """
    Get the appropriate email template for the specified animal type.
    
    Args:
        animal_type: The type of animal detected
        
    Returns:
        str: HTML email template for the animal
    """
    animal_type = animal_type.lower()
    
    # Check for specific animal matches
    for animal in ANIMAL_EMAIL_TEMPLATES:
        if animal in animal_type:
            return ANIMAL_EMAIL_TEMPLATES[animal].format(emergency_contact=EMERGENCY_CONTACT)
    
    # Return default template if no match
    return ANIMAL_EMAIL_TEMPLATES['default'].format(emergency_contact=EMERGENCY_CONTACT)

def attach_image(msg: MIMEMultipart, image_data: Union[str, bytes, BinaryIO], filename: str = "detection.jpg", embed_in_email: bool = True) -> Optional[str]:
    """
    Attach an image to the email message.
    
    Args:
        msg: Email message to attach image to
        image_data: Image data as bytes, file-like object, or path to image
        filename: Filename for the attachment
        embed_in_email: If True, embed image in email body; if False, attach as file
        
    Returns:
        Optional Content-ID for embedded images
    """
    try:
        # Handle different types of image data
        if isinstance(image_data, str):
            # Check if it's a URL
            if image_data.startswith(('http://', 'https://')):
                response = requests.get(image_data, timeout=5)
                response.raise_for_status()
                image_content = response.content
            # Check if it's a base64 encoded string
            elif image_data.startswith('data:image'):
                header, encoded = image_data.split(',', 1)
                image_content = base64.b64decode(encoded)
            # Otherwise, assume it's a file path
            else:
                with open(image_data, 'rb') as f:
                    image_content = f.read()
        elif isinstance(image_data, bytes):
            image_content = image_data
        else:  # File-like object
            image_content = image_data.read()
        
        # Create the image attachment
        image = MIMEImage(image_content)
        
        if embed_in_email:
            # Generate a Content-ID for embedding in HTML
            content_id = f"<image{hash(str(image_content))}>"
            image.add_header('Content-ID', content_id)
            image.add_header('Content-Disposition', 'inline')
            msg.attach(image)
            return content_id
        else:
            # Attach as regular attachment
            image.add_header('Content-Disposition', f'attachment; filename="{filename}"')
            msg.attach(image)
            return None
            
    except Exception as e:
        logger.error(f"Failed to attach image: {e}")
        return None

def send_email(recipient: str, subject: str, detection_data: Dict) -> bool:
    """
    Send an email notification with animal detection details.
    
    Args:
        recipient: Email recipient
        subject: Email subject
        detection_data: Data about the animal detection
        
    Returns:
        success: True if email was sent successfully
    """
    try:
        if not all([EMAIL_USERNAME, EMAIL_PASSWORD, recipient]):
            logger.warning("Missing email credentials or recipient")
            return False
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = EMAIL_USERNAME
        msg['To'] = recipient
        msg['Subject'] = subject
        
        # Get animal details - FIXED TO CHECK BOTH FIELD NAMES
        # First try 'detection_label' (from detection_handler.py)
        animal_type = detection_data.get('detection_label', '')
        
        # If empty, try 'type' (from warning_system.py)
        if not animal_type:
            animal_type = detection_data.get('type', '')
            
        # If still empty, use default
        if not animal_type:
            animal_type = 'unknown'
            
        # Get animal specific precautions using existing templates
        animal_email_content = get_animal_email_template(animal_type)
        
        # Format timestamp in day-month-year format
        timestamp_str = ""
        timestamp = detection_data.get('timestamp', datetime.now())
        
        # Check for formatted_date first (new field)
        if 'formatted_date' in detection_data:
            timestamp_str = detection_data['formatted_date']
        else:
            # Check for existing formatted timestamp
            if 'formatted_timestamp' in detection_data:
                timestamp_str = detection_data['formatted_timestamp']
                # Try to convert to day-month-year format
                try:
                    if isinstance(timestamp_str, str) and '-' in timestamp_str:
                        date_parts = timestamp_str.split(' ')[0].split('-')
                        if len(date_parts) == 3:
                            year, month, day = date_parts
                            time_parts = ' '.join(timestamp_str.split(' ')[1:])
                            timestamp_str = f"{day}-{month}-{year} {time_parts}"
                except Exception as e:
                    logger.error(f"Error formatting timestamp: {e}")
            elif isinstance(timestamp, datetime):
                timestamp_str = format_date_dmy(timestamp)
            else:
                timestamp_str = str(timestamp)
        
        # Extract date and time from timestamp string
        date_display = ""
        time_display = ""
        if timestamp_str:
            try:
                # Try to split timestamp into date and time components
                ts_parts = timestamp_str.split(' ')
                if len(ts_parts) >= 2:
                    date_display = ts_parts[0]  # DD-MM-YYYY
                    time_display = ' '.join(ts_parts[1:])  # HH:MM:SS AM/PM
                else:
                    # If can't split properly, use the whole string
                    date_display = timestamp_str
            except Exception as e:
                logger.error(f"Error splitting timestamp: {e}")
                date_display = timestamp_str
        
        # Get camera name and location info
        camera_name = detection_data.get('camera_name', 'Unknown location')
        location_link = detection_data.get('google_maps_link', '')
        
        # Format location information
        location_info = ""
        if location_link:
            location_info = f"""
            <div style="margin: 25px 0; background-color: #f5f5f5; border-radius: 6px; padding: 20px; border-left: 5px solid #2ecc71;">
                <h3 style="color: #2c3e50; margin-top: 0; margin-bottom: 15px;">📍 Location Details</h3>
                <p>The detection occurred at <strong>{camera_name}</strong>.</p>
                <p style="margin-bottom: 0;"><a href="{location_link}" style="background-color: #2ecc71; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; display: inline-block; font-weight: bold;">View on Google Maps</a></p>
            </div>
            """
        
        # Get screenshot data and try to embed it
        screenshot_url = detection_data.get('screenshot_url', '')
        embedded_image_cid = None
        
        # Check for screenshot data in different formats and attempt to embed it
        try:
            if 'screenshot_data' in detection_data:
                # Direct image data (bytes or base64)
                image_data = detection_data['screenshot_data']
                embedded_image_cid = attach_image(msg, image_data, embed_in_email=True)
                logger.info(f"Embedded image from direct data with CID: {embedded_image_cid}")
            elif 'screenshot_path' in detection_data:
                # Local file path
                image_path = detection_data['screenshot_path']
                embedded_image_cid = attach_image(msg, image_path, embed_in_email=True)
                logger.info(f"Embedded image from file path with CID: {embedded_image_cid}")
            elif screenshot_url:
                # URL to image - try to download and embed
                try:
                    response = requests.get(screenshot_url, timeout=5)
                    response.raise_for_status()
                    
                    # Create the image attachment
                    image = MIMEImage(response.content)
                    
                    # Generate a Content-ID for embedding
                    content_id = f"<image{hash(screenshot_url)}>"
                    image.add_header('Content-ID', content_id)
                    image.add_header('Content-Disposition', 'inline')
                    msg.attach(image)
                    embedded_image_cid = content_id
                    logger.info(f"Successfully embedded image from URL with CID: {embedded_image_cid}")
                except Exception as e:
                    logger.error(f"Failed to embed image from URL: {e}")
                    embedded_image_cid = None
        except Exception as e:
            logger.error(f"Error preparing image attachment: {e}")
            embedded_image_cid = None
        
        # Basic text version (no HTML)
        text_content = f"""
        WildEye Alert: {animal_type} detected at {camera_name}
        Date: {date_display}
        Time: {time_display}
        
        {animal_type} has been detected near your camera at {camera_name}.
        
        PRECAUTIONS:
        - Stay indoors and secure all doors and windows
        - Do NOT approach the animal
        - Contact authorities if needed
        
        {"Location: " + location_link if location_link else ""}
        
        This is an automated message from your WildEye system. Please check your dashboard for more details.
        """
        
        # Get severity if available, otherwise set as low
        severity = detection_data.get('severity', 'low')
        
        # HTML version with improvements
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
              .timestamp {{ display: flex; justify-content: space-between; flex-wrap: wrap; margin-bottom: 15px; }}
              .timestamp-item {{ flex: 1; min-width: 120px; }}
              .view-button {{ display: inline-block; background-color: #3498db; color: white; padding: 10px 20px; 
                              text-decoration: none; border-radius: 4px; margin-top: 15px; font-weight: bold; }}
              .view-button:hover {{ background-color: #2980b9; }}
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
              
              <div class="severity-{severity}">
                <h3 style="margin-top: 0;">Alert Details</h3>
                
                <p><strong>Animal Detected:</strong> {animal_type.title()}</p>
                
                <div class="timestamp">
                  <div class="timestamp-item">
                    <strong>Date:</strong> {date_display}
                  </div>
                  <div class="timestamp-item">
                    <strong>Time:</strong> {time_display}
                  </div>
                </div>
                
                <p style="margin-bottom: 15px;"><strong>Severity:</strong> {severity.upper()}</p>
                
                <a href="#" class="view-button">View Full Details</a>
              </div>
              
              {location_info}
              
              <div class="image-container">
                <h3 style="color: #2c3e50;">📸 Detection Image</h3>
                {f'''
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); background-color: #fff;">
                  <img src="cid:{embedded_image_cid[1:-1]}" alt="Detection Screenshot" style="max-width:100%; border-radius: 4px; display: block; margin: 0 auto;">
                  <div style="text-align: center; margin-top: 10px;">
                    <a href="#" style="background-color: #3498db; color: white; padding: 6px 12px; text-decoration: none; border-radius: 4px; display: inline-block;">View Full Image</a>
                  </div>
                </div>
                ''' if embedded_image_cid else f'''
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); background-color: #fff;">
                  <img src="{screenshot_url}" alt="Detection Screenshot" style="max-width:100%; border-radius: 4px; display: block; margin: 0 auto;">
                  <div style="text-align: center; margin-top: 10px;">
                    <a href="{screenshot_url}" style="background-color: #3498db; color: white; padding: 6px 12px; text-decoration: none; border-radius: 4px; display: inline-block;">View Full Image</a>
                  </div>
                </div>
                ''' if screenshot_url else '<p>No image available for this detection.</p>'}
              </div>
              
              <div class="precautions">
                <h3 style="color: #2c3e50;">⚠️ Recommended Precautions</h3>
                {animal_email_content}
              </div>
              
              <div class="footer">
                <p>This is an automated message from the WildEye system. Please do not reply to this email.</p>
                <p>To manage your notification settings, please log in to your WildEye account.</p>
              </div>
            </div>
          </body>
        </html>
        """
        
        # Attach parts
        part1 = MIMEText(text_content, 'plain')
        part2 = MIMEText(html_body, 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        # Connect to server and send email
        with smtplib.SMTP(EMAIL_SERVER, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"Email notification about {animal_type} sent to {recipient}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False
        
        # Attach parts
        part1 = MIMEText(text_content, 'plain')
        part2 = MIMEText(html_body, 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        # Connect to server and send email
        with smtplib.SMTP(EMAIL_SERVER, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"Email notification about {animal_type} sent to {recipient}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False