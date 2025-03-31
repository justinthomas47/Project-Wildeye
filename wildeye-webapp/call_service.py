# call_service.py
import logging
import os
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Say
from typing import Dict, Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)  # Fixed the logger name from _name_

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
    
    # Check if this is a broadcast alert (from another user's camera)
    is_broadcast = detection_data.get('is_broadcast', False)
    
    # Get distance information for broadcast alerts
    distance_text = "nearby"
    if is_broadcast:
        # Use distance_km if available (numeric value)
        if 'distance_km' in detection_data:
            distance = detection_data['distance_km']
            # Format distance for speech
            if distance < 1:
                distance_text = f"{int(distance * 1000)} meters"
            else:
                distance_text = f"{distance:.1f} kilometers" if distance < 10 else f"{int(distance)} kilometers"
        # Otherwise use distance (might be pre-formatted)
        elif 'distance' in detection_data:
            distance_text = f"{detection_data['distance']} kilometers"
    
    # Generate appropriate script based on animal type and whether it's a broadcast alert
    if is_broadcast:
        # This is a broadcast alert (from another user's camera)
        if animal_type.lower() == 'tiger' or 'tiger' in animal_type.lower():
            script = f"""
            <speak>
                <prosody rate="slow">
                <emphasis level="strong">Urgent Wildlife Alert!</emphasis> 
                <break time="500ms"/>
                This is a broadcast alert from Wild Eye.
                <break time="500ms"/>
                A tiger has been detected {distance_text} from your location.
                <break time="500ms"/>
                This is a high severity detection from another Wild Eye user's camera.
                <break time="500ms"/>
                Please take necessary precautions and remain alert.
                <break time="500ms"/>
                Contact local authorities if you spot the animal.
                <break time="1s"/>
                <emphasis level="strong">Urgent Wildlife Alert!</emphasis> Tiger detected {distance_text} from your location.
                </prosody>
            </speak>
            """
        elif animal_type.lower() == 'leopard' or 'leopard' in animal_type.lower():
            script = f"""
            <speak>
                <prosody rate="slow">
                <emphasis level="strong">Urgent Wildlife Alert!</emphasis> 
                <break time="500ms"/>
                This is a broadcast alert from Wild Eye.
                <break time="500ms"/>
                A leopard has been detected {distance_text} from your location.
                <break time="500ms"/>
                This is a high severity detection from another Wild Eye user's camera.
                <break time="500ms"/>
                Please take necessary precautions and remain alert.
                <break time="500ms"/>
                Contact local authorities if you spot the animal.
                <break time="1s"/>
                <emphasis level="strong">Urgent Wildlife Alert!</emphasis> Leopard detected {distance_text} from your location.
                </prosody>
            </speak>
            """
        elif animal_type.lower() == 'bear' or 'bear' in animal_type.lower():
            script = f"""
            <speak>
                <prosody rate="slow">
                <emphasis level="strong">Urgent Wildlife Alert!</emphasis> 
                <break time="500ms"/>
                This is a broadcast alert from Wild Eye.
                <break time="500ms"/>
                A bear has been detected {distance_text} from your location.
                <break time="500ms"/>
                This is a high severity detection from another Wild Eye user's camera.
                <break time="500ms"/>
                Please take necessary precautions and remain alert.
                <break time="500ms"/>
                Contact local authorities if you spot the animal.
                <break time="1s"/>
                <emphasis level="strong">Urgent Wildlife Alert!</emphasis> Bear detected {distance_text} from your location.
                </prosody>
            </speak>
            """
        elif animal_type.lower() == 'elephant' or 'elephant' in animal_type.lower():
            script = f"""
            <speak>
                <prosody rate="slow">
                <emphasis level="moderate">Wildlife Alert!</emphasis> 
                <break time="500ms"/>
                This is a broadcast alert from Wild Eye.
                <break time="500ms"/>
                An elephant has been detected {distance_text} from your location.
                <break time="500ms"/>
                This detection is from another Wild Eye user's camera in your area.
                <break time="500ms"/>
                Please be aware and take appropriate precautions.
                <break time="1s"/>
                <emphasis level="moderate">Wildlife Alert!</emphasis> Elephant detected {distance_text} from your location.
                </prosody>
            </speak>
            """
        elif animal_type.lower() == 'wild buffalo' or 'wild buffalo' in animal_type.lower():
            script = f"""
            <speak>
                <prosody rate="slow">
                <emphasis level="moderate">Wildlife Alert!</emphasis> 
                <break time="500ms"/>
                This is a broadcast alert from Wild Eye.
                <break time="500ms"/>
                A wild buffalo has been detected {distance_text} from your location.
                <break time="500ms"/>
                This detection is from another Wild Eye user's camera in your area.
                <break time="500ms"/>
                Please be aware and take appropriate precautions.
                <break time="1s"/>
                <emphasis level="moderate">Wildlife Alert!</emphasis> Wild buffalo detected {distance_text} from your location.
                </prosody>
            </speak>
            """
        elif animal_type.lower() == 'wild boar' or 'wild boar' in animal_type.lower():
            script = f"""
            <speak>
                <prosody rate="slow">
                <emphasis level="moderate">Wildlife Alert!</emphasis> 
                <break time="500ms"/>
                This is a broadcast alert from Wild Eye.
                <break time="500ms"/>
                A wild boar has been detected {distance_text} from your location.
                <break time="500ms"/>
                This detection is from another Wild Eye user's camera in your area.
                <break time="500ms"/>
                Please be aware and take appropriate precautions.
                <break time="1s"/>
                <emphasis level="moderate">Wildlife Alert!</emphasis> Wild boar detected {distance_text} from your location.
                </prosody>
            </speak>
            """
        else:
            script = f"""
            <speak>
                <prosody rate="slow">
                <emphasis level="moderate">Wildlife Alert!</emphasis> 
                <break time="500ms"/>
                This is a broadcast alert from Wild Eye.
                <break time="500ms"/>
                A {animal_type} has been detected {distance_text} from your location.
                <break time="500ms"/>
                This detection is from another Wild Eye user's camera in your area.
                <break time="500ms"/>
                Please be aware and take appropriate precautions.
                <break time="1s"/>
                <emphasis level="moderate">Wildlife Alert!</emphasis> {animal_type} detected {distance_text} from your location.
                </prosody>
            </speak>
            """
    else:
        # This is a direct detection from the user's own camera
        if animal_type.lower() == 'tiger' or 'tiger' in animal_type.lower():
            script = f"""
            <speak>
                <prosody rate="slow">
                <emphasis level="strong">Wildlife Alert!</emphasis> Tiger detected at {camera_name}.
                <break time="500ms"/>
                Stay indoors. Secure property. Do not approach.
                <break time="500ms"/>
                Contact forest department immediately.
                </prosody>
            </speak>
            """
        elif animal_type.lower() == 'leopard' or 'leopard' in animal_type.lower():
            script = f"""
            <speak>
                <prosody rate="slow">
                <emphasis level="strong">Wildlife Alert!</emphasis> Leopard detected at {camera_name}.
                <break time="500ms"/>
                Stay indoors. Secure property. Do not approach.
                <break time="500ms"/>
                Contact forest department immediately.
                </prosody>
            </speak>
            """
        elif animal_type.lower() == 'lion' or 'lion' in animal_type.lower():
            script = f"""
            <speak>
                <prosody rate="slow">
                <emphasis level="strong">Wildlife Alert!</emphasis> Lion detected at {camera_name}.
                <break time="500ms"/>
                Stay indoors. Secure property. Do not approach.
                <break time="500ms"/>
                Contact forest department immediately.
                </prosody>
            </speak>
            """
        elif animal_type.lower() == 'bear' or 'bear' in animal_type.lower():
            script = f"""
            <speak>
                <prosody rate="slow">
                <emphasis level="strong">Wildlife Alert!</emphasis> Bear detected at {camera_name}.
                <break time="500ms"/>
                Stay indoors. Secure food sources. Do not approach.
                <break time="500ms"/>
                Contact forest department immediately.
                </prosody>
            </speak>
            """
        elif animal_type.lower() == 'wild buffalo' or 'wild buffalo' in animal_type.lower():
            script = f"""
            <speak>
                <prosody rate="slow">
                <emphasis level="strong">Wildlife Alert!</emphasis> Wild buffalo detected at {camera_name}.
                <break time="500ms"/>
                Stay indoors. Keep far away. Contact authorities immediately.
                </prosody>
            </speak>
            """
        elif animal_type.lower() == 'elephant' or 'elephant' in animal_type.lower():
            script = f"""
            <speak>
                <prosody rate="slow">
                <emphasis level="moderate">Wildlife Warning!</emphasis> Elephant detected at {camera_name}.
                <break time="500ms"/>
                Stay indoors. Remain quiet. Avoid sudden movements.
                <break time="500ms"/>
                Contact authorities if the animal approaches.
                </prosody>
            </speak>
            """
        elif animal_type.lower() == 'wild boar' or 'wild boar' in animal_type.lower():
            script = f"""
            <speak>
                <prosody rate="slow">
                <emphasis level="moderate">Wildlife Warning!</emphasis> Wild boar detected at {camera_name}.
                <break time="500ms"/>
                Keep pets and children away. Contact authorities if aggressive.
                </prosody>
            </speak>
            """
        else:
            script = f"""
            <speak>
                <prosody rate="slow">
                Wildlife Alert. {animal_type} detected at {camera_name}.
                <break time="500ms"/>
                Check Wild Eye app for details.
                </prosody>
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
        
        # Check if it's a broadcast alert and log
        is_broadcast = detection_data.get('is_broadcast', False)
        if is_broadcast:
            distance_text = "unknown distance"
            if 'distance_km' in detection_data:
                distance = detection_data['distance_km']
                if distance < 1:
                    distance_text = f"{int(distance * 1000)} meters"
                else:
                    distance_text = f"{distance:.1f} kilometers" if distance < 10 else f"{int(distance)} kilometers"
            elif 'distance' in detection_data:
                distance_text = f"{detection_data['distance']} kilometers"
                
            logger.info(f"Processing broadcast call alert with distance: {distance_text}")
        
        # Initialize Twilio client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Generate TwiML for call
        response = VoiceResponse()
        
        # Get call script
        call_script = get_call_script(detection_data)
        
        # Add speech with SSML
        response.append(Say(call_script, voice='Polly.Joanna', language='en-US'))
        
        # Make the call with direct TwiML
        call = client.calls.create(
            to=recipient,
            from_=TWILIO_PHONE_NUMBER,
            twiml=str(response)
        )
        
        # Log with appropriate message
        if is_broadcast:
            logger.info(f"Voice broadcast call initiated to {recipient}, SID: {call.sid}")
        else:
            logger.info(f"Voice call notification initiated to {recipient}, SID: {call.sid}")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to make voice call: {e}")
        return False

def create_call_webhook():
    """
    Create a TwiML response for incoming calls or webhooks.
    This is used if someone calls the Twilio number directly.
    
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
    
    # Check if this is a broadcast alert - for broadcast alerts, we might want to be more selective
    is_broadcast = detection_data.get('is_broadcast', False)
    
    # For broadcast alerts, we might want to be more selective
    if is_broadcast:
        # Only call for high severity broadcast alerts by default
        if severity != 'high' and call_threshold != 'all':
            return False
            
        # If it's a priority 1 animal, call regardless of threshold
        if determine_call_priority(animal_type) == 1:
            return True
            
        # Otherwise follow normal threshold rules
    
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