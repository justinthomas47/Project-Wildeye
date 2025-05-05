# process_stream.py (refactored to include detection_handler functionality)
import cv2
from ultralytics import YOLO
import yt_dlp
import threading
from queue import Queue, Empty, Full
import concurrent.futures
import numpy as np
from typing import Generator, Dict, List, Optional, Tuple
import time
import logging
import traceback
import torch
from datetime import datetime, timedelta, timezone
import os
import sys
import tempfile
import hashlib
import shutil
import io
import firebase_admin
from firebase_admin import credentials, firestore
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# Import warning system functionality
from warning_system import create_warning, get_notification_preferences

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directory for temporary downloaded YouTube videos
TEMP_VIDEO_DIR = os.path.join(tempfile.gettempdir(), "wildeye_videos")
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

# Google Drive Integration
SCOPES = ['https://www.googleapis.com/auth/drive.file']
FOLDER_NAME = 'Wildeye Snapshots'  # Name of the folder in Google Drive
folder_id = None  # Will store the folder ID
drive_service = None

# Suppress YOLO output by redirecting stdout during predictions
class SuppressOutput:
    def __init__(self, suppress=True):
        self.suppress = suppress
        self.original_stdout = None
        self.devnull = None
    
    def __enter__(self):
        if self.suppress:
            self.original_stdout = sys.stdout
            self.devnull = open(os.devnull, 'w')
            sys.stdout = self.devnull
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress:
            sys.stdout = self.original_stdout
            self.devnull.close()

# Cache for recent detections to prevent duplicates
recent_detections = {}
# Cache for downloaded YouTube videos
youtube_cache = {}

# Stats for duplicate detections (to summarize instead of logging each one)
duplicate_stats = {
    'count': 0,
    'last_log_time': 0,
    'by_camera': {}
}

# Function to get Indian Standard Time
def get_indian_time():
    """
    Returns the current time in Indian Standard Time (IST) - UTC+5:30
    Using 12-hour time format with AM/PM
    """
    # Create a timezone object for IST (UTC+5:30)
    ist = timezone(timedelta(hours=5, minutes=30))
    
    # Get current UTC time and convert to IST
    utc_time = datetime.now(timezone.utc)
    ist_time = utc_time.astimezone(ist)
    
    # Format the time in 12-hour format
    ist_time = ist_time.replace(tzinfo=None)  # Remove timezone info for Firestore
    
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

def get_camera(db, camera_id: str) -> Optional[Dict]:
    """Get camera details by ID"""
    try:
        doc = db.collection('cameras').document(camera_id).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        logger.error(f"Error getting camera: {e}")
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

def get_video_hash(url):
    """Generate a hash for a YouTube URL to use as a filename"""
    return hashlib.md5(url.encode()).hexdigest()

def download_youtube_video(url, seek_time=0, max_duration=None):
    """
    Enhanced function to download a YouTube video to a temporary file
    with better quality control and error handling.
    
    Returns: Path to the downloaded video file
    """
    video_hash = get_video_hash(url)
    video_path = os.path.join(TEMP_VIDEO_DIR, f"{video_hash}.mp4")
    
    # Check if video already exists in cache
    if os.path.exists(video_path):
        file_age = time.time() - os.path.getmtime(video_path)
        # Keep videos in cache for up to 24 hours
        if file_age < 86400:  # 24 hours in seconds
            try:
                # Validate the cached file before using it
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    # Read a test frame to check if file is valid
                    ret, _ = cap.read()
                    cap.release()
                    
                    if ret:
                        logger.info(f"Using valid cached video for {url}")
                        return video_path
                    else:
                        logger.warning(f"Cached video exists but cannot be read, re-downloading")
                        os.remove(video_path)
                else:
                    logger.warning(f"Cached video exists but cannot be opened, re-downloading")
                    os.remove(video_path)
            except Exception as e:
                logger.warning(f"Error validating cached video: {e}, re-downloading")
                try:
                    os.remove(video_path)
                except:
                    pass
        else:
            # Remove old cached file
            try:
                os.remove(video_path)
            except Exception as e:
                logger.error(f"Error removing old cached file: {e}")
    
    logger.info(f"Downloading YouTube video: {url}")
    
    # Use a temporary file during download to avoid using partially downloaded files
    temp_video_path = f"{video_path}.downloading"
    
    try:
        # First, try getting the best quality mp4 stream
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best[ext=mp4]/best',
            'outtmpl': temp_video_path,
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True,
            'noplaylist': True,
            # Add retries and continuation
            'retries': 3,
            'fragment_retries': 3,
            'continuedl': True,
            # Force single-part download for frame seeking
            'overwrites': True,
            'nopart': True,
        }
        
        # Add time range if specified
        if seek_time > 0 or max_duration:
            ydl_opts['download_ranges'] = download_range = {}
            download_range['ranges'] = []
            
            start_time = seek_time
            end_time = None if not max_duration else (seek_time + max_duration)
            
            time_range = {'start_time': start_time}
            if end_time:
                time_range['end_time'] = end_time
                
            download_range['ranges'].append(time_range)
        
        download_success = False
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
            # Check if download was successful
            if os.path.exists(temp_video_path):
                # Verify the file is valid
                cap = cv2.VideoCapture(temp_video_path)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    
                    if ret:
                        # Move the temporary file to the final path
                        shutil.move(temp_video_path, video_path)
                        download_success = True
                        logger.info(f"Successfully downloaded high quality video to {video_path}")
                    else:
                        logger.warning(f"Downloaded file cannot be read, will try lower quality")
                else:
                    logger.warning(f"Downloaded file cannot be opened, will try lower quality")
            else:
                logger.warning(f"High quality download failed, will try lower quality")
                
        except Exception as e:
            logger.warning(f"Error during high quality download: {e}, trying lower quality")
        
        # If high quality download failed, try with lower quality
        if not download_success:
            try:
                # Clean up any partial downloads
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                
                # Try with lower quality settings
                ydl_opts['format'] = 'best[ext=mp4][height<=480]/best[height<=480]/best'
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                # Check if download was successful
                if os.path.exists(temp_video_path):
                    # Move the temporary file to the final path
                    shutil.move(temp_video_path, video_path)
                    download_success = True
                    logger.info(f"Successfully downloaded lower quality video to {video_path}")
            except Exception as e:
                logger.error(f"Error during lower quality download: {e}")
                
                # Clean up any partial downloads
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
        
        if download_success and os.path.exists(video_path):
            youtube_cache[url] = video_path
            return video_path
        else:
            logger.error(f"Failed to download video: file not found or invalid")
            return None
            
    except Exception as e:
        logger.error(f"Error downloading YouTube video: {e}")
        
        # Clean up any partial downloads
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except:
                pass
                
        return None

def clean_youtube_cache(max_size_gb=5):
    """Clean up old downloaded videos if cache exceeds maximum size"""
    try:
        # Calculate current cache size
        total_size = 0
        files = []
        
        for file in os.listdir(TEMP_VIDEO_DIR):
            if file.endswith('.mp4'):
                file_path = os.path.join(TEMP_VIDEO_DIR, file)
                file_size = os.path.getsize(file_path)
                files.append((file_path, os.path.getmtime(file_path), file_size))
                total_size += file_size
        
        # Convert to GB
        total_size_gb = total_size / (1024**3)
        
        if total_size_gb > max_size_gb:
            logger.info(f"YouTube cache size ({total_size_gb:.2f} GB) exceeds limit ({max_size_gb} GB). Cleaning up...")
            
            # Sort by modification time (oldest first)
            files.sort(key=lambda x: x[1])
            
            # Delete oldest files until under limit
            for file_path, _, file_size in files:
                if total_size_gb <= max_size_gb:
                    break
                    
                try:
                    os.remove(file_path)
                    logger.info(f"Removed old cached video: {file_path}")
                    
                    # Update size
                    total_size -= file_size
                    total_size_gb = total_size / (1024**3)
                    
                    # Remove from cache dictionary if present
                    for url, path in youtube_cache.items():
                        if path == file_path:
                            del youtube_cache[url]
                            break
                            
                except Exception as e:
                    logger.error(f"Error removing cached file {file_path}: {e}")
    
    except Exception as e:
        logger.error(f"Error cleaning YouTube cache: {e}")

def is_recent_duplicate(camera_id: str, detection_label: str, current_time: datetime) -> bool:
    """
    Check if this detection is a duplicate within the last 5 minutes
    for the same camera and animal type.
    """
    global duplicate_stats
    try:
        cache_key = f"{camera_id}_{detection_label}"
        
        if cache_key in recent_detections:
            last_detection_time = recent_detections[cache_key]
            time_difference = current_time - last_detection_time
            
            if time_difference < timedelta(minutes=5):
                # Update duplicate stats
                duplicate_stats['count'] += 1
                
                # Update per-camera stats
                if camera_id not in duplicate_stats['by_camera']:
                    duplicate_stats['by_camera'][camera_id] = {}
                
                if detection_label not in duplicate_stats['by_camera'][camera_id]:
                    duplicate_stats['by_camera'][camera_id][detection_label] = 0
                
                duplicate_stats['by_camera'][camera_id][detection_label] += 1
                
                # Log summary of duplicates every 30 seconds
                current_time_seconds = time.time()
                if current_time_seconds - duplicate_stats['last_log_time'] > 30:
                    # Log summary of duplicates
                    for cam_id, animals in duplicate_stats['by_camera'].items():
                        animal_counts = ", ".join([f"{animal}: {count}" for animal, count in animals.items()])
                        logger.info(f"Skipped duplicates for camera {cam_id}: {animal_counts}")
                    
                    # Reset stats
                    duplicate_stats['count'] = 0
                    duplicate_stats['by_camera'] = {}
                    duplicate_stats['last_log_time'] = current_time_seconds
                
                return True
        
        # Update the last detection time for this camera/animal pair
        recent_detections[cache_key] = current_time
        cleanup_detection_cache(current_time)
        return False
    except Exception as e:
        logger.error(f"Error checking for duplicate detection: {e}")
        return False

def cleanup_detection_cache(current_time: datetime):
    """Remove cache entries older than 5 minutes to prevent memory growth"""
    try:
        keys_to_remove = []
        for key, timestamp in recent_detections.items():
            if current_time - timestamp > timedelta(minutes=5):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del recent_detections[key]
    except Exception as e:
        logger.error(f"Error cleaning up detection cache: {e}")

def add_detection(db, detection_data: Dict, screenshot: bytes = None) -> Optional[str]:
    """
    Add a new detection record with deduplication and create a warning if needed.
    
    Returns: Detection ID if saved successfully, None if duplicate detected
    """
    try:
        # Use Indian Standard Time
        current_time = get_indian_time()
        
        # Format the date string in day-month-year format for display purposes
        formatted_date = current_time.strftime("%d-%m-%Y %I:%M:%S %p")
        
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
            'formatted_date': formatted_date,  # Add formatted date string
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
            'formatted_date': formatted_date,  # Add formatted date string
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

def get_detections_by_camera(db, camera_id: str, limit: int = 100, user_id: str = None) -> List[Dict]:
    """
    Get recent detections for a specific camera, optionally filtered by user.
    
    Args:
        db: Firestore database instance
        camera_id: ID of the camera to get detections for
        limit: Maximum number of detections to return
        user_id: If provided, only return detections for this user
        
    Returns:
        detections: List of detection dictionaries
    """
    try:
        query = db.collection('detections').where('camera_id', '==', camera_id)
        
        if user_id:
            query = query.where('owner_uid', '==', user_id)
            
        query = query.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
        
        return [doc.to_dict() for doc in query.stream()]
    except Exception as e:
        logger.error(f"Error getting detections: {e}")
        return []

class VideoProcessor:
    def __init__(self, model_path: str, num_workers: int = 4, queue_size: int = 30):
        self.num_workers = num_workers
        self.frame_queue = Queue(maxsize=queue_size)
        self.result_queue = Queue(maxsize=queue_size)
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()  # Flag to pause processing
        self.fps = 30  # Default FPS
        self.frame_time = 1/self.fps  # Time per frame
        self.camera_id = None  # Store camera_id
        self.db = None  # Store database instance
        self.last_processed_frame = None  # Keep the last frame for paused state
        self.detection_count = 0  # Count detections
        self.last_detection_log = 0  # Timestamp of last detection log
        self.prefetch_buffer = Queue(maxsize=30)  # Buffer for prefetched frames
        self.current_video_path = None  # Path to current video file
        self.input_type = None  # Store input type (youtube_link, manual, etc.)
        
        try:
            logger.info(f"Loading YOLO model from {model_path}")
            # Check for CUDA availability
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.models = [YOLO(model_path).to(device) for _ in range(num_workers)]
            logger.info(f"YOLO models loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def pause_processing(self):
        """Pause the processing but keep the stream alive"""
        logger.info("Pausing video processing")
        self.pause_flag.set()

    def resume_processing(self):
        """Resume the processing"""
        logger.info("Resuming video processing")
        self.pause_flag.clear()

    def _frame_prefetcher(self, cap: cv2.VideoCapture) -> None:
        """Enhanced prefetcher with better timing control for YouTube videos"""
        frame_count = 0
        last_frame_time = time.time()
        consecutive_failures = 0
        max_failures = 5
        buffer_target = 15  # Target number of frames to keep in buffer
        
        # Get video FPS for proper timing
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or video_fps > 60:  # Invalid or unreasonable FPS
            video_fps = 30
        target_frame_time = 1.0 / video_fps
        
        is_youtube = self.input_type == "youtube_link"
        
        logger.info(f"Starting frame prefetcher thread (YouTube: {is_youtube}, FPS: {video_fps})")
        
        while not self.stop_flag.is_set():
            try:
                current_buffer_size = self.prefetch_buffer.qsize()
                
                # If buffer is full enough, wait a bit
                if current_buffer_size >= buffer_target:
                    time.sleep(0.1)
                    continue
                
                # Calculate how many frames we need to read
                frames_needed = buffer_target - current_buffer_size
                
                # Batch read frames to fill buffer
                for _ in range(frames_needed):
                    if self.stop_flag.is_set():
                        break
                        
                    # Read frame
                    ret, frame = cap.read()
                    if not ret:
                        consecutive_failures += 1
                        logger.warning(f"Prefetcher: Failed to read frame ({consecutive_failures}/{max_failures})")
                        if consecutive_failures >= max_failures:
                            logger.error("Prefetcher: Too many consecutive failures, breaking")
                            return
                        time.sleep(0.1)
                        break  # Try again later
                    
                    # Reset failures counter 
                    consecutive_failures = 0
                    frame_count += 1
                    
                    # Add frame to prefetch buffer
                    try:
                        # For YouTube videos, apply a slight blur to reduce compression artifacts
                        # This can help YOLO detect objects more reliably
                        if is_youtube:
                            frame = cv2.GaussianBlur(frame, (3, 3), 0)
                        
                        # Use non-blocking put with a short timeout
                        if not self.prefetch_buffer.full():
                            self.prefetch_buffer.put((frame, time.time()), block=False)
                    except Full:
                        # If buffer is full, just continue
                        break
                    except Exception as e:
                        logger.error(f"Prefetcher: Error adding to buffer: {e}")
                        break
                
                # Control timing based on video frame rate
                elapsed = time.time() - last_frame_time
                if elapsed < target_frame_time and frames_needed == 0:
                    time.sleep(min(0.1, target_frame_time - elapsed))  # Cap max sleep time
                
                last_frame_time = time.time()
                    
            except Exception as e:
                logger.error(f"Frame prefetcher error: {e}")
                time.sleep(0.1)  # Prevent tight loop on error
                    
        logger.info("Frame prefetcher thread exiting")
        
    def _frame_producer(self, cap: cv2.VideoCapture) -> None:
        frame_count = 0
        last_frame_time = time.time()
        consecutive_failures = 0
        max_failures = 5  # Maximum number of consecutive failures before breaking
        last_frame = None  # Store the last successfully read frame
        
        # Get video FPS for proper timing
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or video_fps > 60:  # Invalid or unreasonable FPS
            video_fps = 30
        target_frame_time = 1.0 / video_fps
        
        # Adjust frame skip based on input type
        frame_skip = 1  # Process every frame by default
        if self.input_type == "youtube_link":
            # For YouTube, process fewer frames to avoid lag
            frame_skip = 3  # Process every 3rd frame
            logger.info(f"YouTube video detected, processing every {frame_skip}th frame")
        
        # Start prefetcher thread for smoother playback
        prefetcher_thread = threading.Thread(target=self._frame_prefetcher, args=(cap,), daemon=True)
        prefetcher_thread.start()

        while not self.stop_flag.is_set():
            try:
                # If processing is paused and we have a last frame, reuse it at reduced rate
                if self.pause_flag.is_set() and last_frame is not None:
                    # Sleep longer when paused to reduce CPU/GPU load
                    time.sleep(0.5)  
                    
                    # Create a copy of the last frame with a "PAUSED" overlay
                    paused_frame = last_frame.copy()
                    cv2.putText(
                        paused_frame,
                        "STREAM PAUSED",
                        (int(paused_frame.shape[1]/2) - 100, int(paused_frame.shape[0]/2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                    
                    try:
                        # Only put frames if not paused for too long
                        if time.time() - last_frame_time < 10:  # 10 seconds max of paused state
                            # Remove oldest frame if queue is full
                            if self.frame_queue.full():
                                try:
                                    self.frame_queue.get_nowait()
                                except Empty:
                                    pass
                                    
                            self.frame_queue.put((paused_frame, time.time()), timeout=1)
                    except Exception as queue_error:
                        logger.error(f"Queue error (paused): {queue_error}")
                    
                    continue

                # Strict timing control for video playback
                current_time = time.time()
                time_diff = current_time - last_frame_time
                if time_diff < target_frame_time:
                    # Sleep precisely to maintain target frame rate
                    time.sleep(target_frame_time - time_diff)

                # Try to get frame from prefetch buffer first
                try:
                    frame, timestamp = self.prefetch_buffer.get_nowait()
                    ret = True
                except Empty:
                    # Fall back to direct capture if buffer is empty
                    ret, frame = cap.read()
                    timestamp = time.time()
                
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame ({consecutive_failures}/{max_failures})")
                    if consecutive_failures >= max_failures:
                        logger.error("Too many consecutive frame reading failures")
                        break
                    time.sleep(0.1)
                    continue
                
                # Store the last successful frame
                last_frame = frame.copy()
                consecutive_failures = 0  # Reset counter on successful frame read
                last_frame_time = time.time()
                frame_count += 1
                
                # Process frames based on frame_skip setting
                if frame_count % frame_skip != 0:
                    continue

                try:
                    # Remove oldest frame if queue is full
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            pass

                    self.frame_queue.put((frame, timestamp), timeout=1)
                except Exception as queue_error:
                    logger.error(f"Queue error: {queue_error}")
                    continue

            except Exception as e:
                logger.error(f"Frame producer error: {e}")
                break

    def _process_frame(self, worker_id: int, frame_data: tuple) -> tuple:
        frame, timestamp = frame_data
        try:
            # If processing is paused, skip detection and just return the frame
            if self.pause_flag.is_set():
                return frame, [], timestamp

            # Define animal-specific confidence thresholds
            animal_confidence_thresholds = {
                "lion": 0.85,
                "bear": 0.85,
                "wildboar": 0.85,
                "wild buffalo": 0.85,
                "elephant": 0.75,
                "tiger": 0.75,
                "leopard": 0.75
            }
            
            # Default confidence threshold for animals not in the dictionary
            default_confidence = 0.7

            # Perform object detection with suppressed output
            with SuppressOutput(suppress=True):
                results = self.models[worker_id].predict(
                    frame,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    stream=True,
                    conf=0.7,  # Initial lower threshold during detection
                    verbose=False  # Disable verbose output
                )

            processed_frame = frame.copy()
            detections = []

            for result in results:
                for box in result.boxes:
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    label = self.models[worker_id].names[class_id]

                    # Get the confidence threshold for this animal
                    confidence_threshold = animal_confidence_thresholds.get(label.lower(), default_confidence)

                    # Only process if confidence is high enough for this specific animal
                    if confidence > confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append({
                            'detection_label': label,
                            'confidence': confidence * 100  # Convert to percentage
                        })

                        # Draw bounding box and label
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text = f"{label} ({confidence:.2f})"
                        cv2.putText(processed_frame, text, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Store the last processed frame
            self.last_processed_frame = processed_frame.copy()

            # If there are detections and we have camera_id and db, save to database
            if detections and self.camera_id and self.db and not self.pause_flag.is_set():
                try:
                    # Convert the processed frame to JPEG format
                    _, buffer = cv2.imencode('.jpg', processed_frame)
                    screenshot = buffer.tobytes()

                    # Only log detection counts periodically to reduce console spam
                    current_time = time.time()
                    self.detection_count += len(detections)
                    
                    # Log every 30 seconds
                    if current_time - self.last_detection_log > 30:
                        logger.info(f"Camera {self.camera_id}: {self.detection_count} detections processed in the last 30 seconds")
                        self.detection_count = 0
                        self.last_detection_log = current_time

                    # Save each detection with deduplication
                    for detection in detections:
                        # Prepare detection data
                        detection_data = {
                            'camera_id': self.camera_id,
                            'detection_label': detection['detection_label'],
                            'confidence': detection['confidence']
                        }
                        
                        # Use add_detection function
                        detection_id = add_detection(self.db, detection_data, screenshot)
                        
                        if detection_id:
                            # Log only new detections (not duplicates)
                            logger.info(f"New detection: {detection['detection_label']} with {detection['confidence']:.1f}% confidence")
                        
                except Exception as e:
                    logger.error(f"Failed to save detection to database: {e}")
                    logger.error(traceback.format_exc())

            return processed_frame, detections, timestamp
        except Exception as e:
            logger.error(f"Frame processing error: {traceback.format_exc()}")
            return frame, [], timestamp

    def _frame_processor_worker(self, worker_id: int) -> None:
        logger.info(f"Starting worker {worker_id}")
        while not self.stop_flag.is_set():
            try:
                frame_data = self.frame_queue.get(timeout=1.0)
                processed_frame, detections, timestamp = self._process_frame(worker_id, frame_data)

                try:
                    if self.result_queue.full():
                        self.result_queue.get_nowait()

                    self.result_queue.put((processed_frame, detections, timestamp), timeout=1)
                except Exception as queue_error:
                    logger.error(f"Result queue error: {queue_error}")

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    def process_stream(self, input_type: str, input_value: str, seek_time: int = 0, camera_id: str = None, db = None, is_visible: bool = False) -> Generator:
        self.camera_id = camera_id  # Store camera_id
        self.db = db  # Store database instance
        self.input_type = input_type  # Store input type for use in other methods
        
        # Set initial pause state based on visibility
        if not is_visible:
            logger.info(f"Starting stream in paused state (not visible): {input_value}")
            self.pause_flag.set()
        else:
            logger.info(f"Starting stream in active state (visible): {input_value}")
            self.pause_flag.clear()
            
        logger.info(f"Starting stream processing - Type: {input_type}, Value: {input_value}, Seek: {seek_time}")
        cap = None

        try:
            # Handle different input types
            if input_type == "manual":
                # For local camera, input_value should be the camera index
                stream_source = int(input_value)
                logger.info(f"Opening local camera at index {stream_source}")
            elif input_type == "youtube_link":
                logger.info("Processing YouTube link")
                
                # Attempt to download the video first for better performance
                video_path = download_youtube_video(input_value, seek_time)
                
                if video_path and os.path.exists(video_path):
                    logger.info(f"Using downloaded video file: {video_path}")
                    stream_source = video_path
                    self.current_video_path = video_path
                else:
                    # Fall back to streaming if download fails
                    logger.warning("Downloaded video not available, falling back to streaming")
                    ydl_opts = {
                        'format': 'best[ext=mp4]/best',  # Try to get mp4 format for better compatibility
                        'quiet': True,
                        'youtube_include_dash_manifest': False
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(input_value, download=False)
                        stream_source = info['url']
            else:
                # RTSP URL case
                stream_source = input_value

            logger.info(f"Opening video stream from source: {stream_source}")
            cap = cv2.VideoCapture(stream_source)
            
            # For YouTube videos, set a larger buffer size and higher priority
            if input_type == "youtube_link":
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)  # Larger buffer for YouTube
            elif isinstance(stream_source, str) and (stream_source.startswith('rtsp://') or stream_source.startswith('http://')):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Standard buffer for other streams

            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video stream from source: {stream_source}")

            # Get and validate video FPS with better fallback
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0 or self.fps > 60:  # Invalid or unreasonable FPS
                # For YouTube, try to get FPS from the video info if available
                if input_type == "youtube_link" and self.current_video_path:
                    try:
                        # Try to read a few frames and calculate FPS
                        start_time = time.time()
                        frame_count = 0
                        for _ in range(20):  # Read 20 frames for estimation
                            ret, _ = cap.read()
                            if ret:
                                frame_count += 1
                        
                        elapsed = time.time() - start_time
                        if elapsed > 0 and frame_count > 0:
                            measured_fps = frame_count / elapsed
                            self.fps = min(60, measured_fps)  # Cap at 60 FPS
                            logger.info(f"Measured YouTube video FPS: {self.fps}")
                        
                        # Reset position
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    except Exception as e:
                        logger.error(f"Error measuring FPS: {e}")
                        self.fps = 30  # Fallback
                else:
                    self.fps = 30  # Default FPS
                    
            self.frame_time = 1/self.fps
            logger.info(f"Video FPS: {self.fps}")

            # Handle seeking for video files
            if seek_time > 0:
                if input_type == "youtube_link":
                    if self.current_video_path:
                        # For downloaded YouTube videos, seeking works well
                        frame_number = int(seek_time * self.fps)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        logger.info(f"Seeked to frame {frame_number} (time: {seek_time}s)")
                    else:
                        # For streaming YouTube, seeking is less reliable
                        # Try to get a new stream URL with the seek time
                        logger.info(f"Getting new YouTube stream with seek time: {seek_time}s")
                        try:
                            ydl_opts = {
                                'format': 'best[ext=mp4]/best',
                                'quiet': True,
                                'youtube_include_dash_manifest': False
                            }
                            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                info = ydl.extract_info(f"{input_value}&t={seek_time}", download=False)
                                new_stream_source = info['url']
                                
                                # Close existing capture and open new one
                                cap.release()
                                cap = cv2.VideoCapture(new_stream_source)
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)  # Larger buffer for YouTube
                                
                                if not cap.isOpened():
                                    logger.error("Failed to open seeked YouTube stream, falling back")
                                    cap = cv2.VideoCapture(stream_source)
                                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
                        except Exception as e:
                            logger.error(f"Error seeking YouTube stream: {e}, using original stream")
                elif input_type != "manual":
                    # For other file types, normal seeking should work
                    frame_number = int(seek_time * self.fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Reset stop flag
            self.stop_flag.clear()

            # Start frame producer
            producer_thread = threading.Thread(
                target=self._frame_producer,
                args=(cap,),
                daemon=True
            )
            producer_thread.start()

            last_frame_time = time.time()
            frame_delivery_rate = self.fps  # Target frame rate for delivery
            target_frame_time = 1.0 / frame_delivery_rate

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Start worker threads
                futures = [executor.submit(self._frame_processor_worker, worker_id)
                          for worker_id in range(self.num_workers)]

                frame_count = 0
                while not self.stop_flag.is_set():
                    try:
                        processed_frame, detections, timestamp = self.result_queue.get(timeout=0.5)

                        # Control frame rate for consistent playback
                        current_time = time.time()
                        elapsed = current_time - last_frame_time
                        
                        # Strictly enforce frame timing for smoother playback
                        if elapsed < target_frame_time:
                            time.sleep(target_frame_time - elapsed)

                        if processed_frame is not None:
                            # Encode frame as JPEG
                            _, buffer = cv2.imencode('.jpg', processed_frame)
                            frame = buffer.tobytes()
                            yield (b'--frame\r\n'
                                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                            last_frame_time = time.time()
                            frame_count += 1
                            
                            # Periodically log frame delivery rate
                            if frame_count % 30 == 0:
                                delivery_time = time.time() - (current_time - elapsed)
                                actual_fps = 30 / delivery_time
                                logger.info(f"Stream delivery rate: {actual_fps:.2f} fps (target: {frame_delivery_rate})")

                    except Empty:
                        # If queue is empty, yield the last processed frame if available
                        # to maintain a constant frame rate
                        if hasattr(self, 'last_processed_frame') and self.last_processed_frame is not None:
                            try:
                                # Encode and send last frame to maintain continuity
                                _, buffer = cv2.imencode('.jpg', self.last_processed_frame)
                                frame = buffer.tobytes()
                                yield (b'--frame\r\n'
                                      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                            except Exception as e:
                                logger.error(f"Error yielding last frame: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Frame delivery error: {e}")
                        time.sleep(0.1)
                        continue

        except Exception as e:
            logger.error(f"Stream processing error: {traceback.format_exc()}")
            raise
        finally:
            logger.info("Stopping stream processing")
            self.stop_flag.set()
            
            # Clean up prefetch buffer
            while not self.prefetch_buffer.empty():
                try:
                    self.prefetch_buffer.get_nowait()
                except Empty:
                    break
            
            # Clean up threads
            if 'producer_thread' in locals():
                producer_thread.join(timeout=5)
            if 'futures' in locals():
                concurrent.futures.wait(futures, timeout=5)
                
            # Release camera/video capture
            if cap is not None:
                cap.release()
            
            # Clean up YouTube cache if it's getting too large
            if input_type == "youtube_link":
                threading.Thread(target=clean_youtube_cache, daemon=True).start()

    def __del__(self):
        self.stop_flag.set()
        # Clean up CUDA cache if using GPU
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

def test_camera_connection(camera_index):
    """
    Thoroughly test camera connection and provide detailed diagnostic information
    Returns: Tuple (bool, dict) - (success status, diagnostic information)
    """
    diagnostics = {
        "index": camera_index,
        "can_open": False,
        "backend": None,
        "frame_read": False,
        "resolution": None,
        "fps": None,
        "error": None,
        "buffer_size": None
    }
    
    try:
        # Try to open the camera
        cap = cv2.VideoCapture(camera_index)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            diagnostics["error"] = "Failed to open camera"
            return False, diagnostics
            
        diagnostics["can_open"] = True
        diagnostics["backend"] = cap.getBackendName()
        
        # Get current buffer size
        diagnostics["buffer_size"] = cap.get(cv2.CAP_PROP_BUFFERSIZE)
        
        # Try to set a larger buffer size
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        # Try reading multiple frames to ensure stable connection
        frames_read = 0
        max_test_frames = 5
        start_time = time.time()
        
        while frames_read < max_test_frames and (time.time() - start_time) < 2.0:
            ret, frame = cap.read()
            if ret and frame is not None:
                frames_read += 1
                
                if frames_read == 1:
                    # Get camera properties from first successful frame
                    diagnostics["resolution"] = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                    diagnostics["fps"] = int(cap.get(cv2.CAP_PROP_FPS))
        
        diagnostics["frame_read"] = frames_read > 0
        
        if frames_read < max_test_frames:
            diagnostics["error"] = f"Only read {frames_read}/{max_test_frames} frames"
            return False, diagnostics
            
        return True, diagnostics
        
    except Exception as e:
        diagnostics["error"] = str(e)
        return False, diagnostics
        
    finally:
        if 'cap' in locals():
            cap.release()

def get_connected_cameras(verbose=True):
    """
    Detect and test all connected cameras with detailed diagnostics
    Returns: List of dicts containing camera information and diagnostics
    """
    available_cameras = []
    max_cameras_to_check = 10
    
    for i in range(max_cameras_to_check):
        success, diagnostics = test_camera_connection(i)
        
        if success or diagnostics["can_open"]:
            camera_info = {
                "index": i,
                "name": "Default Webcam" if i == 0 else f"Camera {i}",
                "status": "OK" if success else "ERROR",
                **diagnostics
            }
            available_cameras.append(camera_info)
            
            if verbose:
                print(f"\nTesting {camera_info['name']} (Index: {i}):")
                print(f"  Status: {camera_info['status']}")
                print(f"  Backend: {diagnostics['backend']}")
                print(f"  Can Open: {diagnostics['can_open']}")
                print(f"  Frame Read: {diagnostics['frame_read']}")
                print(f"  Resolution: {diagnostics['resolution']}")
                print(f"  FPS: {diagnostics['fps']}")
                print(f"  Buffer Size: {diagnostics['buffer_size']}")
                if diagnostics['error']:
                    print(f"  Error: {diagnostics['error']}")
    
    return available_cameras

# Create global VideoProcessor instance
MODEL_PATH = "best.pt"  # Update this to your model path
video_processor = None

def initialize_processor():
    global video_processor
    try:
        logger.info(f"Initializing VideoProcessor with model: {MODEL_PATH}")
        video_processor = VideoProcessor(MODEL_PATH, num_workers=4)
        logger.info("VideoProcessor initialized successfully")
        
        # Also initialize the Google Drive service
        try:
            init_drive_service()
            logger.info("Google Drive service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive service: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize VideoProcessor: {e}")
        raise

def process_stream(input_type: str, input_value: str, seek_time: int = 0, camera_id: str = None, db = None, is_visible: bool = False) -> Generator:
    """Process a video stream and yield processed frames"""
    global video_processor
    if video_processor is None:
        initialize_processor()

    logger.info(f"Processing stream request - Type: {input_type}, Value: {input_value}, Visible: {is_visible}")
    try:
        yield from video_processor.process_stream(input_type, input_value, seek_time, camera_id, db, is_visible)
    except Exception as e:
        logger.error(f"Stream processing failed: {e}")
        raise

def pause_stream(camera_id: str):
    """Pause processing for a specific camera"""
    global video_processor
    if video_processor is not None:
        logger.info(f"Pausing stream for camera: {camera_id}")
        video_processor.pause_processing()
        return True
    return False

def resume_stream(camera_id: str):
    """Resume processing for a specific camera"""
    global video_processor
    if video_processor is not None:
        logger.info(f"Resuming stream for camera: {camera_id}")
        video_processor.resume_processing()
        return True
    return False

# Run periodic YouTube cache cleanup
def periodic_cache_cleanup():
    """Run periodic cleanup of YouTube cache in a background thread"""
    while True:
        try:
            # Clean cache every hour
            time.sleep(3600)
            clean_youtube_cache()
        except Exception as e:
            logger.error(f"Error in periodic cache cleanup: {e}")

# Start periodic cache cleanup in a daemon thread
cleanup_thread = threading.Thread(target=periodic_cache_cleanup, daemon=True)
cleanup_thread.start()

# Initialize the processor when module is imported
initialize_processor()

# Export the necessary functions
__all__ = ['process_stream', 'get_connected_cameras', 'pause_stream', 'resume_stream', 
           'get_detections_by_camera', 'add_detection', 'init_drive_service']