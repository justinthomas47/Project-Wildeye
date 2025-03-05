# detection_handler_updates.py
# This file contains the updated detection handler that integrates with the warning system

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import firebase_admin
from firebase_admin import credentials, firestore
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io
import logging
import traceback
import os
import time
from warning_system import create_warning, get_notification_preferences

# Configure logging
logger = logging.getLogger(__name__)

# Google Drive settings
SCOPES = ['https://www.googleapis.com/auth/drive.file']
FOLDER_NAME = 'Wildeye Snapshots'
folder_id = None
drive_service = None

# Cache for recent detections to prevent duplicates
recent_detections = {}

def get_indian_time():
    """
    Returns the current time in Indian Standard Time (IST) - UTC+5:30
    """
    # Create a timezone object for IST (UTC+5:30)
    ist = timezone(timedelta(hours=5, minutes=30))
    
    # Get current UTC time and convert to IST
    utc_time = datetime.now(timezone.utc)
    ist_time = utc_time.astimezone(ist)
    
    return ist_time

def init_drive_service():
    """Initialize Google Drive service and create/get folder"""
    global folder_id, drive_service
    try:
        credentials = service_account.Credentials.from_service_account_file(
            'service_account.json', scopes=SCOPES)
        service = build('drive', 'v3', credentials=credentials)
        
        # Check if folder exists
        query = f"name='{FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = service.files().list(q=query, spaces='drive').execute()
        items = results.get('files', [])
        
        if not items:
            # Create folder if it doesn't exist
            folder_metadata = {
                'name': FOLDER_NAME,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = service.files().create(body=folder_metadata, fields='id').execute()
            folder_id = folder.get('id')
        else:
            folder_id = items[0]['id']
            
        drive_service = service
        return service
    except Exception as e:
        logger.error(f"Drive initialization error: {e}")
        return None

def save_screenshot_to_drive(screenshot: bytes, detection_id: str) -> str:
    """Save screenshot to Google Drive"""
    try:
        if not drive_service or not folder_id:
            if not init_drive_service():
                raise Exception("Drive service not initialized")
            
        # Create file metadata
        filename = f"{detection_id}_{int(get_indian_time().timestamp())}.jpg"
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        
        # Create media
        fh = io.BytesIO(screenshot)
        media = MediaIoBaseUpload(fh, mimetype='image/jpeg', resumable=True)
        
        # Upload file
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink'
        ).execute()
        
        # Make file publicly viewable
        permission = {
            'type': 'anyone',
            'role': 'reader'
        }
        drive_service.permissions().create(
            fileId=file['id'],
            body=permission
        ).execute()
        
        return file.get('webViewLink')
    except Exception as e:
        logger.error(f"Error uploading to Google Drive: {e}")
        return None

def save_screenshot_locally(screenshot: bytes, detection_id: str) -> str:
    """Save screenshot to local storage as fallback"""
    try:
        os.makedirs('static/screenshots', exist_ok=True)
        filename = f"static/screenshots/{detection_id}_{int(get_indian_time().timestamp())}.jpg"
        with open(filename, 'wb') as f:
            f.write(screenshot)
        return filename
    except Exception as e:
        logger.error(f"Error saving screenshot locally: {e}")
        return None

def is_recent_duplicate(camera_id: str, detection_label: str, current_time: datetime) -> bool:
    """Check if this detection is a duplicate within the last 5 minutes"""
    try:
        cache_key = f"{camera_id}_{detection_label}"
        
        if cache_key in recent_detections:
            last_detection_time = recent_detections[cache_key]
            time_difference = current_time - last_detection_time
            
            if time_difference < timedelta(minutes=5):
                logger.info(f"Skipping duplicate detection of {detection_label} from camera {camera_id}")
                return True
        
        recent_detections[cache_key] = current_time
        cleanup_detection_cache(current_time)
        return False
    except Exception as e:
        logger.error(f"Error checking for duplicate detection: {e}")
        return False

def cleanup_detection_cache(current_time: datetime):
    """Remove cache entries older than 5 minutes"""
    try:
        keys_to_remove = []
        for key, timestamp in recent_detections.items():
            if current_time - timestamp > timedelta(minutes=5):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del recent_detections[key]
    except Exception as e:
        logger.error(f"Error cleaning up detection cache: {e}")

def get_camera(db, camera_id: str) -> Optional[Dict]:
    """Get camera details by ID"""
    try:
        doc = db.collection('cameras').document(camera_id).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        logger.error(f"Error getting camera: {e}")
        return None

def add_detection(db, detection_data: Dict, screenshot: bytes = None) -> Optional[str]:
    """
    Add a new detection record with deduplication and create a warning if needed.
    
    Returns: Detection ID if saved successfully, None if duplicate detected
    """
    try:
        # Use Indian Standard Time
        current_time = get_indian_time()
        
        # Check for recent duplicate detection
        if is_recent_duplicate(
            detection_data['camera_id'], 
            detection_data['detection_label'],
            current_time
        ):
            return None

        # Get camera details
        camera = get_camera(db, detection_data['camera_id'])
        if not camera:
            raise ValueError(f"Camera {detection_data['camera_id']} not found")

        # Create detection document
        detection_ref = db.collection('detections').document()
        detection_id = detection_ref.id
        
        # Store screenshot if provided
        screenshot_url = None
        if screenshot:
            screenshot_url = save_screenshot_to_drive(screenshot, detection_id)
            if not screenshot_url:
                screenshot_url = save_screenshot_locally(screenshot, detection_id)

        # Prepare detection record
        detection_record = {
            'detection_id': detection_id,
            'camera_id': detection_data['camera_id'],
            'camera_name': camera['camera_name'],
            'google_maps_link': camera.get('google_maps_link', ''),
            'mobile_number': camera.get('mobile_number', ''),
            'detection_label': detection_data['detection_label'],
            'confidence': detection_data.get('confidence', 100.0),
            'timestamp': current_time,
            'screenshot_url': screenshot_url
        }

        # Add the detection to Firestore database
        detection_ref.set(detection_record)
        
        # Also add to detection_logs collection for history display
        log_ref = db.collection('detection_logs').document()
        log_data = {
            'id': log_ref.id,
            'animal': detection_data['detection_label'],
            'camera': camera['camera_name'],
            'location': camera.get('google_maps_link', ''),
            'timestamp': current_time,
            'confidence': detection_data.get('confidence', 100.0),
            'image_url': screenshot_url
        }
        log_ref.set(log_data)
        
        # Create a warning for this detection
        try:
            # Get notification preferences
            notification_preferences = get_notification_preferences(db)
            
            # Create warning
            warning_id = create_warning(db, detection_record, notification_preferences)
            if warning_id:
                logger.info(f"Created warning {warning_id} for detection {detection_id}")
        except Exception as warning_error:
            logger.error(f"Error creating warning: {warning_error}")
            logger.error(traceback.format_exc())
        
        logger.info(f"Added new detection: {detection_data['detection_label']} from {camera['camera_name']}")
        return detection_id
    except Exception as e:
        logger.error(f"Error adding detection: {e}")
        logger.error(traceback.format_exc())
        return None

def get_detections_by_camera(db, camera_id: str, limit: int = 100) -> List[Dict]:
    """Get recent detections for a specific camera"""
    try:
        detections = (db.collection('detections')
                     .where('camera_id', '==', camera_id)
                     .order_by('timestamp', direction=firestore.Query.DESCENDING)
                     .limit(limit)
                     .stream())
        return [doc.to_dict() for doc in detections]
    except Exception as e:
        logger.error(f"Error getting detections: {e}")
        return []

def get_detections_by_timerange(db, start_time: datetime, end_time: datetime, 
                              camera_id: Optional[str] = None) -> List[Dict]:
    """Get detections within a time range, optionally filtered by camera"""
    try:
        query = (db.collection('detections')
                .where('timestamp', '>=', start_time)
                .where('timestamp', '<=', end_time)
                .order_by('timestamp', direction=firestore.Query.DESCENDING))
        
        if camera_id:
            query = query.where('camera_id', '==', camera_id)
            
        detections = query.stream()
        return [doc.to_dict() for doc in detections]
    except Exception as e:
        logger.error(f"Error getting detections by timerange: {e}")
        return []

# Initialize Drive service when module is imported
try:
    drive_service = init_drive_service()
except Exception as e:
    logger.error(f"Failed to initialize Drive service: {e}")