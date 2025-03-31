# call_service.py
import logging
import os
import uuid
from typing import Dict, Optional
from datetime import datetime
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from twilio.base.exceptions import TwilioRestException

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', '')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER', '')
WEBHOOK_BASE_URL = os.environ.get('WEBHOOK_BASE_URL', 'http://localhost:5000')
EMERGENCY_CONTACT = os.environ.get('EMERGENCY_CONTACT', 'Forest Dept: XXX-XXX-XXXX')

# Call configuration
MAX_CALL_ATTEMPTS = int(os.environ.get('MAX_CALL_ATTEMPTS', 2))  # Number of times to repeat the message

# In-memory session store for call data
session_store = {}

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

def should_call_for_detection(detection_data: Dict, preferences: Dict) -> bool:
    """
    Determine if a call should be made based on detection severity and user preferences.
    
    Args:
        detection_data: Data about the animal detection
        preferences: User notification preferences
        
    Returns:
        bool: True if a call should be made
    """
    # If call notifications are disabled, don't call
    if not preferences.get('call', {}).get('enabled', False):
        return False
    
    # Get detection severity
    severity = detection_data.get('severity', 'low')
    
    # Get user's call threshold (default to 'high' if not specified)
    threshold = preferences.get('call', {}).get('threshold', 'high')
    
    # Check if this is a broadcast alert - only call for high severity broadcasts 
    is_broadcast = detection_data.get('is_broadcast', False)
    if is_broadcast and severity != 'high':
        logger.info(f"Not calling for broadcast alert with severity {severity} (threshold: high for broadcasts)")
        return False
    
    # Check severity against threshold
    if threshold == 'all':
        # Call for all detections
        return True
    elif threshold == 'medium':
        # Call for medium and high severity
        return severity in ['medium', 'high']
    else:
        # Default: call only for high severity
        return severity == 'high'

def make_call(recipient: str, detection_data: Dict) -> bool:
    """
    Make a voice call notification.
    
    Args:
        recipient: Phone number to call
        detection_data: Data about the animal detection
        
    Returns:
        success: True if call was made successfully
    """
    try:
        # Validate requirements
        if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
            logger.error("Missing Twilio credentials")
            return False
            
        if not TWILIO_PHONE_NUMBER:
            logger.error("Missing Twilio phone number")
            return False
            
        if not recipient:
            logger.error("No recipient phone number provided")
            return False
        
        # Format phone number with country code if needed
        if not recipient.startswith('+'):
            recipient = '+' + recipient.lstrip('0')
            
        logger.info(f"Preparing to make call to {recipient}")
        
        # Get detection details
        animal_type = detection_data.get('detection_label', '') or detection_data.get('type', 'unknown animal')
        camera_name = detection_data.get('camera_name', 'unknown location')
        severity = detection_data.get('severity', 'medium')
        
        # Check if this is a broadcast alert
        is_broadcast = detection_data.get('is_broadcast', False)
        
        # Get distance for broadcast alerts
        distance = None
        distance_text = ""
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
            
            if distance_text:
                logger.info(f"Processing broadcast call alert with distance: {distance_text}")
        
        # Store call details in session storage for TwiML
        call_data = {
            'animal_type': animal_type,
            'camera_name': camera_name,
            'severity': severity,
            'is_broadcast': is_broadcast,
            'timestamp': datetime.now().strftime("%I:%M %p"),
            'call_attempts': 0,
            'max_attempts': MAX_CALL_ATTEMPTS
        }
        
        # Add distance if this is a broadcast alert
        if is_broadcast and distance_text:
            call_data['distance'] = distance_text
        
        # Save the call data
        call_id = str(uuid.uuid4())
        session_store[call_id] = call_data
        
        # Set the URL for the call webhook with the call_id parameter
        call_url = f"{WEBHOOK_BASE_URL}/call_webhook?call_id={call_id}"
        
        # Make the call
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            to=recipient,
            from_=TWILIO_PHONE_NUMBER,
            url=call_url,
            status_callback=f"{WEBHOOK_BASE_URL}/call_status_callback",
            status_callback_event=['completed', 'busy', 'no-answer', 'failed'],
            timeout=30
        )
        
        logger.info(f"Call initiated to {recipient}, SID: {call.sid}")
        return True
        
    except TwilioRestException as e:
        logger.error(f"Twilio error making call: {e}")
        return False
    except Exception as e:
        logger.error(f"Error making call notification: {e}")
        return False

def create_call_webhook(request=None):
    """
    Create TwiML response for voice calls.
    Generates the voice message for the wildlife detection.
    
    Args:
        request: Flask request object (optional)
        
    Returns:
        twiml: TwiML response for the call
    """
    try:
        # Initialize TwiML response
        response = VoiceResponse()
        
        # Get call ID from request parameters
        call_id = None
        if request and request.args.get('call_id'):
            call_id = request.args.get('call_id')
        elif request and request.form.get('call_id'):
            call_id = request.form.get('call_id')
            
        # If we have a call ID, try to get the call data
        if call_id and call_id in session_store:
            call_data = session_store[call_id]
            
            # Update call attempts counter
            call_data['call_attempts'] += 1
            session_store[call_id] = call_data
            
            # Extract call data
            animal_type = call_data.get('animal_type', 'unknown animal')
            camera_name = call_data.get('camera_name', 'unknown location')
            severity = call_data.get('severity', 'medium')
            is_broadcast = call_data.get('is_broadcast', False)
            timestamp = call_data.get('timestamp', 'recent time')
            distance = call_data.get('distance', 'unknown distance')
            
            # Create appropriate message based on severity and broadcast status
            if is_broadcast:
                # Broadcast alert - animal detected by another user's camera
                if severity == 'high':
                    message = f"""
                    <speak>
                        <emphasis level="strong">Urgent Wildlife Alert!</emphasis>
                        <break time="500ms"/>
                        This is a broadcast alert from Wild Eye wildlife detection system.
                        <break time="500ms"/>
                        A {animal_type} has been detected {distance} from your location at {timestamp}.
                        <break time="500ms"/>
                        This is a high severity detection from another Wild Eye user's camera in your area.
                        <break time="500ms"/>
                        Please take necessary precautions and remain alert.
                        <break time="500ms"/>
                        Contact local authorities if you spot the animal.
                        <break time="1s"/>
                        This message will repeat once.
                        <break time="1s"/>
                        <emphasis level="strong">Urgent Wildlife Alert!</emphasis>
                        <break time="500ms"/>
                        A {animal_type} has been detected {distance} from your location.
                        <break time="500ms"/>
                        Please take necessary precautions.
                    </speak>
                    """
                else:
                    message = f"""
                    <speak>
                        <emphasis level="moderate">Wildlife Alert!</emphasis>
                        <break time="500ms"/>
                        This is a broadcast alert from Wild Eye wildlife detection system.
                        <break time="500ms"/>
                        A {animal_type} has been detected {distance} from your location at {timestamp}.
                        <break time="500ms"/>
                        This detection is from another Wild Eye user's camera in your area.
                        <break time="500ms"/>
                        Please be aware and take appropriate precautions.
                        <break time="1s"/>
                        This message will repeat once.
                        <break time="1s"/>
                        <emphasis level="moderate">Wildlife Alert!</emphasis>
                        <break time="500ms"/>
                        A {animal_type} has been detected {distance} from your location.
                        <break time="500ms"/>
                        Please be aware and take appropriate precautions.
                    </speak>
                    """
            else:
                # Direct alert - animal detected by your own camera
                if severity == 'high':
                    message = f"""
                    <speak>
                        <emphasis level="strong">Urgent Wildlife Alert!</emphasis>
                        <break time="500ms"/>
                        This is an alert from Wild Eye wildlife detection system.
                        <break time="500ms"/>
                        A {animal_type} has been detected at {camera_name} at {timestamp}.
                        <break time="500ms"/>
                        This is a high severity detection.
                        <break time="500ms"/>
                        Please take necessary precautions and remain alert.
                        <break time="500ms"/>
                        Contact local authorities if needed.
                        <break time="1s"/>
                        This message will repeat once.
                        <break time="1s"/>
                        <emphasis level="strong">Urgent Wildlife Alert!</emphasis>
                        <break time="500ms"/>
                        A {animal_type} has been detected at {camera_name}.
                        <break time="500ms"/>
                        Please take necessary precautions.
                    </speak>
                    """
                else:
                    message = f"""
                    <speak>
                        <emphasis level="moderate">Wildlife Alert!</emphasis>
                        <break time="500ms"/>
                        This is an alert from Wild Eye wildlife detection system.
                        <break time="500ms"/>
                        A {animal_type} has been detected at {camera_name} at {timestamp}.
                        <break time="500ms"/>
                        Please be aware and take appropriate precautions.
                        <break time="1s"/>
                        This message will repeat once.
                        <break time="1s"/>
                        <emphasis level="moderate">Wildlife Alert!</emphasis>
                        <break time="500ms"/>
                        A {animal_type} has been detected at {camera_name}.
                        <break time="500ms"/>
                        Please be aware and take appropriate precautions.
                    </speak>
                    """
            
            # Add the message to the response
            response.say(message, voice='Polly.Aditi')
            
            # If maximum attempts not reached, try again
            if call_data['call_attempts'] < call_data['max_attempts']:
                # Add a redirect to try again
                response.redirect(f"/call_webhook?call_id={call_id}", method='GET')
            else:
                # Clean up the session data
                if call_id in session_store:
                    del session_store[call_id]
        else:
            # If no call data found, use a generic message
            generic_message = """
            <speak>
                <emphasis level="moderate">Wildlife Alert!</emphasis>
                <break time="500ms"/>
                This is an alert from Wild Eye wildlife detection system.
                <break time="500ms"/>
                An animal has been detected in your monitoring area.
                <break time="500ms"/>
                Please check your Wild Eye application for more details.
            </speak>
            """
            response.say(generic_message, voice='Polly.Aditi')
            
        return str(response)
    except Exception as e:
        logger.error(f"Error generating call TwiML: {e}")
        # Return a simple TwiML in case of error
        response = VoiceResponse()
        response.say("Wildlife alert from Wild Eye system. Please check your application for details.", voice='Polly.Aditi')
        return str(response)

def cleanup_call_sessions():
    """
    Clean up old call sessions that may be lingering in memory.
    Should be called periodically by a background task.
    """
    try:
        # Get current timestamp
        now = datetime.now()
        expired_keys = []
        
        # Find and collect expired sessions (older than 1 hour)
        for call_id, call_data in session_store.items():
            # If the session has a timestamp, check if it's old
            if 'created_at' in call_data:
                created_at = call_data['created_at']
                if isinstance(created_at, datetime):
                    age_seconds = (now - created_at).total_seconds()
                    if age_seconds > 3600:  # 1 hour
                        expired_keys.append(call_id)
            else:
                # If no timestamp, assume it's old and should be cleaned up
                expired_keys.append(call_id)
                
        # Remove expired sessions
        for call_id in expired_keys:
            if call_id in session_store:
                del session_store[call_id]
                
        logger.info(f"Cleaned up {len(expired_keys)} expired call sessions")
        
    except Exception as e:
        logger.error(f"Error cleaning up call sessions: {e}")