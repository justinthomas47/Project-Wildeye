#models.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import firebase_admin
from firebase_admin import credentials, firestore
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io
import logging
import cv2
import torch
import traceback
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.file']
FOLDER_NAME = 'Wildeye Snapshots'  # Name of the folder in Google Drive
folder_id = None  # Will store the folder ID

def init_drive_service():
    """Initialize Google Drive service and create/get folder"""
    global folder_id
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
            
        return service
    except Exception as e:
        logger.error(f"Drive initialization error: {e}")
        return None

drive_service = init_drive_service()

# Cache for recent detections to prevent duplicates
recent_detections = {}

def init_db():
    """Initialize Firebase database"""
    try:
        cred = credentials.Certificate("firebase_admin_key.json")
        firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        logger.error(f"Firebase initialization error: {e}")
        return None

def save_screenshot_to_drive(screenshot: bytes, detection_id: str) -> str:
    """Save screenshot to Google Drive"""
    try:
        if not drive_service or not folder_id:
            raise Exception("Drive service not initialized")
            
        # Create file metadata
        filename = f"{detection_id}_{int(datetime.now().timestamp())}.jpg"
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
        filename = f"static/screenshots/{detection_id}_{int(datetime.now().timestamp())}.jpg"
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

# Camera Collection Operations
def add_camera(db, camera_data: Dict) -> str:
    """Add a new camera to the database"""
    try:
        camera_ref = db.collection('cameras').document()
        camera_data['camera_id'] = camera_ref.id
        camera_ref.set(camera_data)
        return camera_ref.id
    except Exception as e:
        logger.error(f"Error adding camera: {e}")
        raise

def update_camera(db, camera_id: str, updates: Dict) -> bool:
    """Update camera details"""
    try:
        db.collection('cameras').document(camera_id).update(updates)
        return True
    except Exception as e:
        logger.error(f"Error updating camera: {e}")
        return False

def get_camera(db, camera_id: str) -> Optional[Dict]:
    """Get camera details by ID"""
    try:
        doc = db.collection('cameras').document(camera_id).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        logger.error(f"Error getting camera: {e}")
        return None

def add_detection(db, detection_data: Dict, screenshot: bytes = None) -> Optional[str]:
    """Add a new detection record with deduplication"""
    try:
        current_time = datetime.now()
        
        if is_recent_duplicate(
            detection_data['camera_id'], 
            detection_data['detection_label'],
            current_time
        ):
            logger.info(f"Skipping duplicate detection of {detection_data['detection_label']} "
                       f"from camera {detection_data['camera_id']}")
            return None

        camera = get_camera(db, detection_data['camera_id'])
        if not camera:
            raise ValueError(f"Camera {detection_data['camera_id']} not found")

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
            'detection_label': detection_data['detection_label'],
            'confidence': detection_data.get('confidence', 100.0),  # Add confidence value
            'timestamp': current_time,
            'screenshot_url': screenshot_url
        }

        # Add to detections collection
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
        
        return detection_id
    except Exception as e:
        logger.error(f"Error adding detection: {e}")
        raise

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