# email_service.py
import logging
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import requests
from datetime import datetime
from typing import Dict, Optional, List, Tuple

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

def attach_image_from_url(msg: MIMEMultipart, image_url: str, embed_in_email: bool = True) -> Optional[Tuple[str, MIMEImage]]:
    """
    Attach an image from a URL to the email message.
    
    Args:
        msg: Email message to attach image to
        image_url: URL of the image to attach
        embed_in_email: If True, embed image in email body; if False, attach as file
        
    Returns:
        Optional tuple with Content-ID and MIMEImage object for embedded images
    """
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        
        # Create the image attachment
        image = MIMEImage(response.content)
        
        if embed_in_email:
            # Generate a Content-ID for embedding in HTML
            content_id = f"<image{hash(image_url)}>"
            image.add_header('Content-ID', content_id)
            image.add_header('Content-Disposition', 'inline')
            msg.attach(image)
            return content_id, image
        else:
            # Attach as regular attachment
            filename = image_url.split('/')[-1]
            if '?' in filename:  # Remove query parameters
                filename = filename.split('?')[0]
            if not filename or len(filename) < 3:
                filename = "detection.jpg"
            
            image.add_header('Content-Disposition', f'attachment; filename="{filename}"')
            msg.attach(image)
            return None
            
    except Exception as e:
        logger.error(f"Failed to attach image from URL {image_url}: {e}")
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
        
        # Get camera name and location info
        camera_name = detection_data.get('camera_name', 'Unknown location')
        location_link = detection_data.get('google_maps_link', '')
        
        # Get screenshot URL
        screenshot_url = detection_data.get('screenshot_url', '')
        embedded_image_cid = None
        
        # Prepare for image embedding if screenshot is available
        if screenshot_url:
            result = attach_image_from_url(msg, screenshot_url, embed_in_email=True)
            if result:
                embedded_image_cid, _ = result
        
        # Basic text version (no HTML)
        text_content = f"""
        WildEye Alert: {animal_type} detected at {camera_name}
        Time: {timestamp_str}
        
        {animal_type} has been detected near your camera at {camera_name}.
        
        PRECAUTIONS:
        - Stay indoors and secure all doors and windows
        - Do NOT approach the animal
        - Contact authorities if needed
        
        {"Location: " + location_link if location_link else ""}
        
        This is an automated message from your WildEye system. Please check your dashboard for more details.
        """
        
        # HTML version
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #4f46e5; color: white; padding: 10px 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
                .alert-high {{ color: #ff0000; font-weight: bold; }}
                .alert-medium {{ color: #ff9900; font-weight: bold; }}
                .alert-low {{ color: #0066cc; font-weight: bold; }}
                .details {{ margin: 15px 0; }}
                .actions {{ background-color: #e6f7ff; padding: 15px; border-left: 4px solid #1890ff; margin: 10px 0; }}
                img.detection-image {{ max-width: 100%; border: 1px solid #ddd; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>WildEye Detection Alert</h2>
                </div>
                <div class="content">
                    <p><span class="alert-high">Alert:</span> {animal_type} detected at {camera_name}</p>
                    
                    <div class="details">
                        <p><strong>Time:</strong> {timestamp_str}</p>
                        {f'<p><strong>Location:</strong> <a href="{location_link}">View on map</a></p>' if location_link else ''}
                    </div>
                    
                    {animal_email_content}
                    
                    <div class="actions">
                        <p><strong>Actions:</strong></p>
                        <p>Login to your WildEye dashboard to:</p>
                        <ul>
                            <li>View the detection</li>
                            <li>Acknowledge this alert</li>
                            <li>Mark as resolved when appropriate</li>
                        </ul>
                    </div>
                    
                    {f'<p><img src="cid:{embedded_image_cid[1:-1]}" alt="Detection Screenshot" class="detection-image"></p>' if embedded_image_cid else ''}
                </div>
                <div class="footer">
                    <p>This is an automated message from your WildEye system. Do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Attach parts
        part1 = MIMEText(text_content, 'plain')
        part2 = MIMEText(html_content, 'html')
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