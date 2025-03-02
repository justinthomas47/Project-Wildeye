# detection_handler.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Generator, Tuple
import firebase_admin
from firebase_admin import credentials, firestore
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io
import logging
import cv2
import torch
from ultralytics import YOLO
import traceback
import os
import time
import threading
from queue import Queue, Empty
import concurrent.futures
import numpy as np

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
drive_service = None

# Cache for recent detections to prevent duplicates
recent_detections = {}

# Stats for duplicate detections (to summarize instead of logging each one)
duplicate_stats = {
    'count': 0,
    'last_log_time': 0,
    'by_camera': {}
}

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
    Add a new detection record with deduplication.
    
    Returns: Detection ID if saved successfully, None if duplicate detected
    """
    try:
        current_time = datetime.now()
        
        # Check for recent duplicate detection
        if is_recent_duplicate(
            detection_data['camera_id'], 
            detection_data['detection_label'],
            current_time
        ):
            # We no longer log individual skipped duplicates - handled in is_recent_duplicate
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
        
        logger.info(f"Added new detection: {detection_data['detection_label']} from {camera['camera_name']}")
        return detection_id
    except Exception as e:
        logger.error(f"Error adding detection: {e}")
        logger.error(traceback.format_exc())
        return None

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

    def _frame_producer(self, cap: cv2.VideoCapture) -> None:
        frame_count = 0
        last_frame_time = time.time()
        consecutive_failures = 0
        max_failures = 5  # Maximum number of consecutive failures before breaking
        last_frame = None  # Store the last successfully read frame

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

                # Maintain proper timing for normal operation
                current_time = time.time()
                time_diff = current_time - last_frame_time
                if time_diff < self.frame_time:
                    time.sleep(self.frame_time - time_diff)

                ret, frame = cap.read()
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
                
                # Process every other frame to reduce load
                if frame_count % 2 != 0:
                    continue

                try:
                    # Remove oldest frame if queue is full
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            pass

                    self.frame_queue.put((frame, time.time()), timeout=1)
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

            # Perform object detection
            results = self.models[worker_id].predict(
                frame,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                stream=True,
                conf=0.5,
                verbose=False  # Disable verbose output
            )

            processed_frame = frame.copy()
            detections = []

            for result in results:
                for box in result.boxes:
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    label = self.models[worker_id].names[class_id]

                    # Only process if confidence is high enough
                    if confidence > 0.5:
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

                    # Save each detection
                    for detection in detections:
                        detection_data = {
                            'camera_id': self.camera_id,
                            'detection_label': detection['detection_label'],
                            'confidence': detection['confidence'],
                            'timestamp': datetime.now()
                        }
                        
                        # Call add_detection which handles deduplication internally
                        detection_id = add_detection(self.db, detection_data, screenshot)
                        # Logging for new detections is handled inside add_detection
                            
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
                import yt_dlp
                ydl_opts = {
                    'format': 'best[ext=mp4]',
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
            
            # Add buffer size for network streams
            if isinstance(stream_source, str) and (stream_source.startswith('rtsp://') or stream_source.startswith('http://')):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video stream from source: {stream_source}")

            # Get and validate video FPS
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0 or self.fps > 60:  # Invalid or unreasonable FPS
                self.fps = 30
            self.frame_time = 1/self.fps
            logger.info(f"Video FPS: {self.fps}")

            # Handle seeking for video files
            if seek_time > 0 and input_type != "manual":
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

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Start worker threads
                futures = [executor.submit(self._frame_processor_worker, worker_id)
                          for worker_id in range(self.num_workers)]

                while not self.stop_flag.is_set():
                    try:
                        processed_frame, detections, timestamp = self.result_queue.get(timeout=1.0)

                        # Control frame rate
                        current_time = time.time()
                        elapsed = current_time - last_frame_time
                        if elapsed < self.frame_time:
                            time.sleep(self.frame_time - elapsed)

                        if processed_frame is not None:
                            # Encode frame as JPEG
                            _, buffer = cv2.imencode('.jpg', processed_frame)
                            frame = buffer.tobytes()
                            yield (b'--frame\r\n'
                                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                            last_frame_time = time.time()

                    except Empty:
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
            
            # Clean up threads
            if 'producer_thread' in locals():
                producer_thread.join(timeout=5)
            if 'futures' in locals():
                concurrent.futures.wait(futures, timeout=5)
                
            # Release camera/video capture
            if cap is not None:
                cap.release()

    def __del__(self):
        self.stop_flag.set()
        # Clean up CUDA cache if using GPU
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

# Create global VideoProcessor instance
MODEL_PATH = "best.pt"  # Update this to your model path
video_processor = None

def initialize_processor():
    global video_processor
    try:
        logger.info(f"Initializing VideoProcessor with model: {MODEL_PATH}")
        video_processor = VideoProcessor(MODEL_PATH, num_workers=4)
        logger.info("VideoProcessor initialized successfully")
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

# Initialize drive service when module is imported
init_drive_service()