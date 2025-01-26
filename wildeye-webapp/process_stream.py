import cv2
from ultralytics import YOLO
import yt_dlp
import threading
from queue import Queue, Empty
import concurrent.futures
import numpy as np
from typing import Optional, Generator
import time
import logging
import socket
import traceback

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, model_path: str, num_workers: int = 4, queue_size: int = 30):
        self.num_workers = num_workers
        self.frame_queue = Queue(maxsize=queue_size)
        self.result_queue = Queue(maxsize=queue_size)
        self.stop_flag = threading.Event()
        
        try:
            logger.info(f"Loading YOLO model from {model_path}")
            self.models = [YOLO(model_path).to('cuda') for _ in range(num_workers)]
            logger.info("YOLO models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
        
    def _frame_producer(self, stream_source: str) -> None:
        logger.info(f"Attempting to open stream: {stream_source}")
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries and not self.stop_flag.is_set():
            try:
                cap = cv2.VideoCapture(stream_source)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open stream (attempt {retry_count + 1}/{max_retries})")
                
                logger.info("Stream opened successfully")
                
                while not self.stop_flag.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Failed to read frame")
                        break
                        
                    try:
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except Empty:
                                pass
                        
                        self.frame_queue.put(frame, timeout=1)
                    except Exception as queue_error:
                        logger.error(f"Queue error: {queue_error}")
                    
                break
                
            except Exception as e:
                logger.error(f"Stream connection error: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying connection in 2 seconds...")
                    time.sleep(2)
                else:
                    logger.error("Max retries reached. Failed to establish stream connection.")
                    return
                    
            finally:
                if 'cap' in locals():
                    cap.release()
                    
    def _process_frame(self, worker_id: int, frame: np.ndarray) -> tuple:
        try:
            results = self.models[worker_id].predict(
                frame, 
                device='cuda',
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
                    
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{label} ({confidence:.2f})"
                    cv2.putText(processed_frame, text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
            return processed_frame, detections
        except Exception as e:
            logger.error(f"Frame processing error: {traceback.format_exc()}")
            return frame, []
            
    def _frame_processor_worker(self, worker_id: int) -> None:
        logger.info(f"Starting worker {worker_id}")
        while not self.stop_flag.is_set():
            try:
                frame = self.frame_queue.get(timeout=1.0)
                processed_frame, detections = self._process_frame(worker_id, frame)
                
                try:
                    if self.result_queue.full():
                        self.result_queue.get_nowait()
                    
                    self.result_queue.put((processed_frame, detections), timeout=1)
                except Exception as queue_error:
                    logger.error(f"Result queue error: {queue_error}")
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    def process_stream(self, input_type: str, input_value: str) -> Generator:
        logger.info(f"Starting stream processing - Type: {input_type}")
        
        try:
            if input_type == "youtube_link":
                logger.info("Processing YouTube link")
                ydl_opts = {
                    'format': 'bestvideo',
                    'noplaylist': True,
                    'quiet': True,
                    'outtmpl': '-',
                    'force-ipv4': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(input_value, download=False)
                    stream_source = info_dict.get("url", input_value)
            else:
                stream_source = input_value
                
            logger.info(f"Final stream source: {stream_source}")
            
            producer_thread = threading.Thread(
                target=self._frame_producer,
                args=(stream_source,),
                daemon=True
            )
            producer_thread.start()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self._frame_processor_worker, worker_id) 
                           for worker_id in range(self.num_workers)]
                    
                try:
                    while not self.stop_flag.is_set():
                        try:
                            processed_frame, _ = self.result_queue.get(timeout=1.0)
                            _, buffer = cv2.imencode('.jpg', processed_frame)
                            frame = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        except Empty:
                            time.sleep(0.1)
                            continue
                        except Exception as e:
                            logger.error(f"Frame delivery error: {e}")
                            time.sleep(0.5)
                            
                finally:
                    logger.info("Stopping stream processing")
                    self.stop_flag.set()
                    producer_thread.join(timeout=5)
                    concurrent.futures.wait(futures, timeout=5)
                    
        except Exception as e:
            logger.error(f"Stream processing error: {traceback.format_exc()}")
            raise
            
    def __del__(self):
        self.stop_flag.set()

# Create global VideoProcessor instance with error handling
try:
    MODEL_PATH = r"C:\Users\JUSTIN THOMAS\Desktop\Project Final code\runs\detect\wild_animal_detection_model8\weights\best.pt"
    logger.info(f"Initializing VideoProcessor with model: {MODEL_PATH}")
    video_processor = VideoProcessor(MODEL_PATH, num_workers=4)
except Exception as e:
    logger.error(f"Failed to initialize VideoProcessor: {e}")
    raise

def process_stream(input_type: str, input_value: str) -> Generator:
    """Wrapper function with error handling"""
    logger.info(f"Processing stream request - Type: {input_type}, Value: {input_value}")
    try:
        return video_processor.process_stream(input_type, input_value)
    except Exception as e:
        logger.error(f"Stream processing failed: {e}")
        raise