# call_service.py
import logging
import os
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Say
from typing import Dict, Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables or config
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', '')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER', '')
TWILIO_TWIML_APP_SID = os.environ.get('TWILIO_TWIML_APP_SID', '')  # For TwiML apps
BASE_URL = os.environ.get('BASE_URL', 'https://your-app-domain.com')  # For callbacks
EMERGENCY_CONTACT = os.environ.get('EMERGENCY_CONTACT', 'Forest Department')

# Mapping of animal types to call priority (1-3, 1 is highest)
ANIMAL_CALL_PRIORITY = {
    'tiger': 1,
    'leopard': 1,
    'lion': 1,
    'bear': 1,
    'elephant': 2,
    'wild buffalo': 2,
    'wild boar': 3
}

def determine_call_priority(detection_label: str) -> int:
    """
    Determine the priority of the call based on the detection label.
    Priority 1: Immediate danger (tigers, leopards, etc.)
    Priority 2: Potential danger
    Priority 3: Lower risk
    
    Args:
        detection_label: Type of animal detected
        
    Returns:
        priority: 1, 2, or 3 (1 is highest priority)
    """
    detection_label = detection_label.lower()
    
    # Check for specific animals
    for animal, priority in ANIMAL_CALL_PRIORITY.items():
        if animal in detection_label:
            return priority
    
    # Default to medium priority if not found
    return 3

def get_call_script(detection_data: Dict) -> str:
    """
    Generate a concise call script based on the detection data.
    
    Args:
        detection_data: Dictionary containing detection information
        
    Returns:
        script: Text to be spoken during the call
    """
    # Get animal type from either 'detection_label' or 'type' field
    animal_type = detection_data.get('detection_label', '') or detection_data.get('type', 'unknown animal')
    camera_name = detection_data.get('camera_name', 'your camera')
    
    # Format timestamp
    timestamp = "recently"
    if 'formatted_date' in detection_data:
        timestamp = detection_data['formatted_date']
    elif 'formatted_timestamp' in detection_data:
        timestamp = detection_data['formatted_timestamp']
    
    # Generate appropriate script based on animal type, matching with email templates
    if 'tiger' in animal_type.lower():
        script = f"""
        <speak>
            <emphasis level="strong">Wildlife Alert!</emphasis> Tiger detected at {camera_name}.
            <break time="300ms"/>
            Stay indoors. Secure property. Do not approach.
            <break time="300ms"/>
            Contact forest department immediately.
        </speak>
        """
    elif 'leopard' in animal_type.lower():
        script = f"""
        <speak>
            <emphasis level="strong">Wildlife Alert!</emphasis> Leopard detected at {camera_name}.
            <break time="300ms"/>
            Stay indoors. Secure property. Do not approach.
            <break time="300ms"/>
            Contact forest department immediately.
        </speak>
        """
    elif 'lion' in animal_type.lower():
        script = f"""
        <speak>
            <emphasis level="strong">Wildlife Alert!</emphasis> Lion detected at {camera_name}.
            <break time="300ms"/>
            Stay indoors. Secure property. Do not approach.
            <break time="300ms"/>
            Contact forest department immediately.
        </speak>
        """
    elif 'bear' in animal_type.lower():
        script = f"""
        <speak>
            <emphasis level="strong">Wildlife Alert!</emphasis> Bear detected at {camera_name}.
            <break time="300ms"/>
            Stay indoors. Secure food sources. Do not approach.
            <break time="300ms"/>
            Contact forest department immediately.
        </speak>
        """
    elif 'wild buffalo' in animal_type.lower():
        script = f"""
        <speak>
            <emphasis level="strong">Wildlife Alert!</emphasis> Wild buffalo detected at {camera_name}.
            <break time="300ms"/>
            Stay indoors. Keep far away. Contact authorities immediately.
        </speak>
        """
    elif 'elephant' in animal_type.lower():
        script = f"""
        <speak>
            <emphasis level="moderate">Wildlife Warning!</emphasis> Elephant detected at {camera_name}.
            <break time="300ms"/>
            Stay indoors. Remain quiet. Avoid sudden movements.
            <break time="300ms"/>
            Contact authorities if the animal approaches.
        </speak>
        """
    elif 'wild boar' in animal_type.lower():
        script = f"""
        <speak>
            <emphasis level="moderate">Wildlife Warning!</emphasis> Wild boar detected at {camera_name}.
            <break time="300ms"/>
            Keep pets and children away. Contact authorities if aggressive.
        </speak>
        """
    else:
        script = f"""
        <speak>
            Wildlife Alert. {animal_type} detected at {camera_name}.
            <break time="300ms"/>
            Check Wild Eye app for details.
        </speak>
        """
    
    return script

def make_call(recipient: str, detection_data: Dict) -> bool:
    """
    Make a voice call to alert about wildlife detection.
    
    Args:
        recipient: Phone number to call
        detection_data: Data about the animal detection
        
    Returns:
        success: True if call was initiated successfully
    """
    try:
        if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, recipient]):
            logger.warning("Missing Twilio credentials or recipient phone number")
            return False
        
        # Initialize Twilio client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Generate TwiML for call
        response = VoiceResponse()
        
        # Get call script
        call_script = get_call_script(detection_data)
        
        # Add speech with SSML
        response.append(Say(call_script, voice='Polly.Joanna', language='en-US'))
        
        # If you need advanced call flow, you can use callbacks instead
        # Make the call
        call = client.calls.create(
            to=recipient,
            from_=TWILIO_PHONE_NUMBER,
            twiml=str(response),
            # For asynchronous status updates (optional)
            # status_callback=f"{BASE_URL}/call_status_callback",
            # status_callback_event=['initiated', 'ringing', 'answered', 'completed'],
            # status_callback_method='POST'
        )
        
        logger.info(f"Voice call notification initiated to {recipient}, SID: {call.sid}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to make voice call: {e}")
        return False

def create_call_webhook():
    """
    Create a TwiML response for incoming calls or webhooks.
    
    Returns:
        str: TwiML response as string
    """
    response = VoiceResponse()
    response.say(
        "This is the Wild Eye wildlife detection system. "
        "If you received a call from this number, it was regarding "
        "a wildlife detection alert in your area. "
        "Please check the Wild Eye application for more details.",
        voice='Polly.Joanna'
    )
    return str(response)

def should_call_for_detection(detection_data: Dict, preferences: Dict) -> bool:
    """
    Determine if a call should be made for this detection based on severity and preferences.
    
    Args:
        detection_data: Detection data
        preferences: User notification preferences
        
    Returns:
        bool: True if a call should be made
    """
    # Check if calls are enabled in user preferences
    if not preferences.get('call', {}).get('enabled', False):
        return False
    
    # Get animal type
    animal_type = detection_data.get('detection_label', '') or detection_data.get('type', '')
    
    # Get detection severity
    severity = detection_data.get('severity', 'low')
    
    # Get call threshold from preferences (default to high)
    call_threshold = preferences.get('call', {}).get('threshold', 'high')
    
    # Determine if a call should be made based on severity
    if call_threshold == 'all':
        # Call for all detections
        return True
    elif call_threshold == 'medium' and severity in ['high', 'medium']:
        # Call for medium and high severity
        return True
    elif call_threshold == 'high' and severity == 'high':
        # Call only for high severity
        return True
    
    # If none of the above, also check animal priority
    priority = determine_call_priority(animal_type)
    if priority == 1 and call_threshold in ['high', 'medium']:
        # Always call for priority 1 animals if threshold is medium or high
        return True
    
    # Default to not calling
    return False