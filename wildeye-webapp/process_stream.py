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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, model_path: str, num_workers: int = 4, queue_size: int = 30):
        self.num_workers = num_workers
        self.frame_queue = Queue(maxsize=queue_size)
        self.result_queue = Queue(maxsize=queue_size)
        self.stop_flag = threading.Event()
        self.fps = 30  # Default FPS
        self.frame_time = 1/self.fps  # Time per frame

        try:
            logger.info(f"Loading YOLO model from {model_path}")
            # Check for CUDA availability
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.models = [YOLO(model_path).to(device) for _ in range(num_workers)]
            logger.info(f"YOLO models loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def _frame_producer(self, cap: cv2.VideoCapture) -> None:
        frame_count = 0
        last_frame_time = time.time()
        consecutive_failures = 0
        max_failures = 5  # Maximum number of consecutive failures before breaking

        while not self.stop_flag.is_set():
            try:
                # Maintain proper timing
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
                
                consecutive_failures = 0  # Reset counter on successful frame read
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
                    last_frame_time = time.time()
                except Exception as queue_error:
                    logger.error(f"Queue error: {queue_error}")
                    continue

            except Exception as e:
                logger.error(f"Frame producer error: {e}")
                break

    def _process_frame(self, worker_id: int, frame_data: tuple) -> tuple:
        frame, timestamp = frame_data
        try:
            # Perform object detection
            results = self.models[worker_id].predict(
                frame,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                stream=True,
                conf=0.5
            )

            processed_frame = frame.copy()
            detections = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    label = self.models[worker_id].names[class_id]

                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'label': label
                    })

                    # Draw bounding box and label
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{label} ({confidence:.2f})"
                    cv2.putText(processed_frame, text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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

    def process_stream(self, input_type: str, input_value: str, seek_time: int = 0) -> Generator:
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

def process_stream(input_type: str, input_value: str, seek_time: int = 0) -> Generator:
    """Process a video stream and yield processed frames"""
    global video_processor
    if video_processor is None:
        initialize_processor()

    logger.info(f"Processing stream request - Type: {input_type}, Value: {input_value}")
    try:
        yield from video_processor.process_stream(input_type, input_value, seek_time)
    except Exception as e:
        logger.error(f"Stream processing failed: {e}")
        raise

def get_connected_cameras():
    """
    Detect and return a list of connected cameras, excluding the built-in webcam
    Returns: List of dicts containing camera information
    """
    available_cameras = []
    max_cameras_to_check = 10  # Check first 10 indexes
    
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Read a test frame
            ret, _ = cap.read()
            if ret:
                # Get camera details
                backend = cap.getBackendName()
                camera_name = f"External Camera {i}"
                
                # Skip built-in webcam (usually index 0)
                if i != 0:
                    available_cameras.append({
                        "index": i,
                        "name": camera_name,
                        "backend": backend
                    })
            cap.release()
    
    return available_cameras

# Initialize the processor when module is imported
initialize_processor()

# Export the necessary functions
__all__ = ['process_stream', 'get_connected_cameras']