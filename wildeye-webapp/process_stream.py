import cv2
from ultralytics import YOLO
import yt_dlp
import threading
from queue import Queue, Empty
import concurrent.futures
import numpy as np
from typing import Generator
import time
import logging
import traceback
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import sys

# Import from detection_handler module
from detection_handler import add_detection, initialize_processor as init_detection_handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

            # Perform object detection with suppressed output
            with SuppressOutput(suppress=True):
                results = self.models[worker_id].predict(
                    frame,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    stream=True,
                    conf=0.7,
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
                    if confidence > 0.7:
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

                    # Save each detection with deduplication using detection_handler
                    for detection in detections:
                        # Prepare detection data
                        detection_data = {
                            'camera_id': self.camera_id,
                            'detection_label': detection['detection_label'],
                            'confidence': detection['confidence']
                        }
                        
                        # Use add_detection from detection_handler module
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
        
        # Also initialize the detection handler
        try:
            init_detection_handler()
            logger.info("Detection handler initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize detection handler: {e}")
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

# Initialize the processor when module is imported
initialize_processor()

# Export the necessary functions
__all__ = ['process_stream', 'get_connected_cameras', 'pause_stream', 'resume_stream']