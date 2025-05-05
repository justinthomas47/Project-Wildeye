# youtube_handler.py - Revised with better error handling, timing controls, and playback rate management
import os
import time
import threading
import tempfile
import logging
import subprocess
import shutil
import cv2
import yt_dlp
import traceback
import numpy as np
from urllib.parse import urlparse, parse_qs
from queue import Queue, Empty, Full

# Configure logging
logger = logging.getLogger(__name__)

class YouTubeDownloader:
    def __init__(self, cache_dir=None):
        """Initialize the YouTube downloader with an optional cache directory."""
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), 'wildeye_videos')
        self.downloads = {}  # Track downloads: {video_id: {path, last_accessed, status}}
        self.cleanup_interval = 3600  # Cleanup unused videos every hour
        self.lock = threading.Lock()
        self.frame_buffer = {}  # Buffer for frames by video_id
        self.buffer_size = 30  # Number of frames to keep in buffer
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Start cleanup thread
        self.stop_cleanup = threading.Event()
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_videos, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"Initialized YouTubeDownloader with cache at {self.cache_dir}")
    
    def _extract_video_id(self, url):
        """Extract the YouTube video ID from a URL."""
        try:
            # Parse the URL
            parsed_url = urlparse(url)
            
            # YouTube URLs can be in different formats
            if parsed_url.netloc == 'youtu.be':
                # Short URL format: https://youtu.be/VIDEO_ID
                return parsed_url.path.lstrip('/')
            elif parsed_url.netloc in ('www.youtube.com', 'youtube.com'):
                # Regular URL format: https://www.youtube.com/watch?v=VIDEO_ID
                if parsed_url.path == '/watch':
                    query = parse_qs(parsed_url.query)
                    return query.get('v', [None])[0]
                # Embed URL format: https://www.youtube.com/embed/VIDEO_ID
                elif parsed_url.path.startswith('/embed/'):
                    return parsed_url.path.split('/')[2]
                # Shortened URL with t parameter
                elif parsed_url.path.startswith('/shorts/'):
                    return parsed_url.path.split('/')[2]
            
            # Return None if format is not recognized
            return None
        except Exception as e:
            logger.error(f"Error extracting video ID: {e}")
            return None
    
    def _get_direct_stream_url(self, url):
        """Get the direct streaming URL from YouTube."""
        try:
            logger.info(f"Getting direct stream URL for {url}")
            ydl_opts = {
                # Prefer mp4 format for better compatibility with OpenCV
                'format': 'best[ext=mp4][height<=720]/best[ext=mp4]/best',
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                stream_url = info.get('url')
                fps = info.get('fps', 30)  # Get video FPS if available
                logger.info(f"Got direct stream URL with FPS: {fps}")
                return stream_url, fps
        except Exception as e:
            logger.error(f"Error getting direct stream URL: {e}")
            return None, 30  # Default 30 FPS
    
    def _get_cached_video_path(self, video_id, seek_time=0):
        """Get the path to a cached video, download if not available."""
        with self.lock:
            # Check if already downloaded or downloading
            if video_id in self.downloads:
                download_info = self.downloads[video_id]
                
                # If download is complete, update access time and return path
                if download_info['status'] == 'complete':
                    self.downloads[video_id]['last_accessed'] = time.time()
                    return download_info['path'], download_info.get('fps', 30)
                
                # If download is in progress, wait for it
                elif download_info['status'] == 'downloading':
                    logger.info(f"Download for {video_id} already in progress, waiting...")
                    return None, 30  # Caller should check again later
            
            # Start a new download
            self.downloads[video_id] = {
                'path': None,
                'last_accessed': time.time(),
                'status': 'downloading',
                'fps': 30  # Default FPS
            }
        
        try:
            # Define output path
            output_path = os.path.join(self.cache_dir, f"{video_id}.mp4")
            temp_output_path = output_path + '.download'
            
            # Configure youtube-dl options
            ydl_opts = {
                # Prefer lower resolution for better performance and smaller file size
                'format': 'best[ext=mp4][height<=720]/best[ext=mp4]/best',
                'outtmpl': temp_output_path,
                'quiet': True,
                'no_warnings': True,
                'youtube_include_dash_manifest': False,
                'fragment_retries': 10,
                'retries': 10,
                # Allow partial download resume
                'continuedl': True,
            }
            
            # Download the video
            logger.info(f"Downloading YouTube video {video_id} to {temp_output_path}")
            
            video_info = None
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                video_info = ydl.extract_info(video_url, download=True)
            
            # Get video FPS if available
            fps = 30  # Default FPS
            if video_info and 'fps' in video_info:
                fps = video_info['fps']
                logger.info(f"Video FPS from metadata: {fps}")
            
            # Move from temp file to final path
            if os.path.exists(temp_output_path):
                shutil.move(temp_output_path, output_path)
                logger.info(f"Download complete: {output_path}")
                
                # Check if we got the FPS from the metadata, if not try to get it from the video file
                if fps == 30:
                    try:
                        cap = cv2.VideoCapture(output_path)
                        if cap.isOpened():
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            if fps <= 0 or fps > 60:
                                fps = 30  # Use default if invalid
                            else:
                                logger.info(f"Video FPS from file: {fps}")
                        cap.release()
                    except Exception as e:
                        logger.error(f"Error getting FPS from video file: {e}")
                
                # Update download status
                with self.lock:
                    self.downloads[video_id]['path'] = output_path
                    self.downloads[video_id]['status'] = 'complete'
                    self.downloads[video_id]['last_accessed'] = time.time()
                    self.downloads[video_id]['fps'] = fps
                
                return output_path, fps
            else:
                logger.error(f"Download failed: {temp_output_path} not found")
                
                # Update download status
                with self.lock:
                    self.downloads[video_id]['status'] = 'failed'
                
                return None, 30
                
        except Exception as e:
            logger.error(f"Error downloading YouTube video {video_id}: {e}")
            logger.error(traceback.format_exc())
            
            # Update download status
            with self.lock:
                self.downloads[video_id]['status'] = 'failed'
            
            return None, 30
    
    def _prefetch_frames(self, cap, video_id, fps):
        """Prefetch frames into a buffer for smoother playback."""
        if video_id not in self.frame_buffer:
            self.frame_buffer[video_id] = Queue(maxsize=self.buffer_size)
        
        buffer = self.frame_buffer[video_id]
        
        # Calculate target frame time (seconds per frame)
        frame_time = 1.0 / fps
        last_frame_time = time.time()
        
        try:
            while cap.isOpened():
                # Don't fill buffer if it's already almost full
                if buffer.qsize() >= self.buffer_size * 0.8:
                    time.sleep(0.01)
                    continue
                
                # Control frame reading rate to avoid CPU overload
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_time / 2:  # Read faster than playback rate to build buffer
                    time.sleep(max(0.001, (frame_time / 2) - elapsed))
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add to buffer, drop frame if buffer full
                try:
                    buffer.put_nowait((ret, frame))
                except Full:
                    # If buffer is full, just continue
                    pass
                
                last_frame_time = time.time()
        except Exception as e:
            logger.error(f"Error in prefetch_frames: {e}")
        finally:
            logger.info(f"Prefetch thread for {video_id} exiting")
    
    def get_video_stream(self, url, seek_time=0):
        """Get a video stream from a YouTube URL, using cache if available."""
        video_id = self._extract_video_id(url)
        if not video_id:
            logger.error(f"Could not extract video ID from URL: {url}")
            return self._get_direct_stream(url, seek_time)
        
        # Try to get cached path (download if needed)
        try:
            video_path, fps = self._get_cached_video_path(video_id, seek_time)
            if not video_path:
                # If download is in progress or failed, fall back to direct streaming
                logger.info(f"Falling back to direct streaming for {video_id}")
                return self._get_direct_stream(url, seek_time)
            
            # Create a frame buffer for this video if it doesn't exist
            if video_id not in self.frame_buffer:
                self.frame_buffer[video_id] = Queue(maxsize=self.buffer_size)
            
            # Open local video file
            logger.info(f"Opening cached video file: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open cached video file: {video_path}")
                return self._get_direct_stream(url, seek_time)
            
            # Set up a larger buffer for smoother playback
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
            
            # Log video properties
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0 or video_fps > 60:
                video_fps = fps  # Use the FPS from metadata or default
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            logger.info(f"Cached video properties: {width}x{height}, {video_fps} fps, {frame_count} frames")
            
            # Seek to position if requested
            if seek_time > 0:
                frame_pos = int(seek_time * video_fps)
                logger.info(f"Seeking to frame {frame_pos} (time: {seek_time}s)")
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            
            # Start prefetch thread
            prefetch_thread = threading.Thread(
                target=self._prefetch_frames, 
                args=(cap, video_id, video_fps),
                daemon=True
            )
            prefetch_thread.start()
            
            # Return a VideoCapture-like object with rate-limited frame delivery
            return self._create_rate_controlled_capture(cap, video_id, video_fps)
            
        except Exception as e:
            logger.error(f"Error opening cached video: {e}")
            logger.error(traceback.format_exc())
            return self._get_direct_stream(url, seek_time)
    
    def _create_rate_controlled_capture(self, cap, video_id, fps):
        """Create a VideoCapture-like object with rate-limited frame delivery."""
        class RateControlledCapture:
            def __init__(self, cap, video_id, fps, parent):
                self.cap = cap
                self.fps = fps
                self.frame_time = 1.0 / fps
                self.last_frame_time = time.time()
                self.video_id = video_id
                self.parent = parent
                self.is_open = True
                self.last_frame = None
            
            def isOpened(self):
                return self.is_open and self.cap.isOpened()
            
            def read(self):
                if not self.is_open:
                    return False, None
                
                # Check if there are prefetched frames
                if self.video_id in self.parent.frame_buffer:
                    buffer = self.parent.frame_buffer[self.video_id]
                    if not buffer.empty():
                        try:
                            ret, frame = buffer.get_nowait()
                            if ret:
                                self.last_frame = frame.copy()  # Save last valid frame
                                return ret, frame
                        except Empty:
                            pass
                
                # If no prefetched frames or buffer error, read directly
                # Control frame delivery rate for smooth playback
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                
                # Enforce minimum frame time to prevent speeding through video
                if elapsed < self.frame_time * 0.9:  # Allow small tolerance
                    time.sleep(self.frame_time - elapsed)
                
                ret, frame = self.cap.read()
                if ret:
                    self.last_frame = frame.copy()  # Save last valid frame
                    self.last_frame_time = time.time()
                elif self.last_frame is not None:
                    # If we can't read a new frame but have a last frame,
                    # return the last frame to prevent stuttering
                    return True, self.last_frame
                
                return ret, frame
            
            def get(self, prop_id):
                if prop_id == cv2.CAP_PROP_FPS:
                    return self.fps
                return self.cap.get(prop_id)
            
            def set(self, prop_id, value):
                # For seek operations, clear the buffer
                if prop_id == cv2.CAP_PROP_POS_FRAMES or prop_id == cv2.CAP_PROP_POS_MSEC:
                    if self.video_id in self.parent.frame_buffer:
                        try:
                            buffer = self.parent.frame_buffer[self.video_id]
                            while not buffer.empty():
                                buffer.get_nowait()
                        except Exception as e:
                            logger.error(f"Error clearing buffer: {e}")
                
                return self.cap.set(prop_id, value)
            
            def release(self):
                self.is_open = False
                self.cap.release()
                # Clear buffer
                if self.video_id in self.parent.frame_buffer:
                    try:
                        buffer = self.parent.frame_buffer[self.video_id]
                        while not buffer.empty():
                            buffer.get_nowait()
                    except Exception as e:
                        logger.error(f"Error clearing buffer: {e}")
        
        return RateControlledCapture(cap, video_id, fps, self)
    
    def _get_direct_stream(self, url, seek_time=0):
        """Fall back to direct streaming without caching."""
        try:
            logger.info(f"Using direct streaming for URL: {url}")
            
            # Get the direct URL for the video
            stream_url, fps = self._get_direct_stream_url(url)
            if not stream_url:
                logger.error("Failed to get direct stream URL")
                return self._get_fallback_stream(url)
            
            # For direct streams with seek times, try to inject the seek parameter
            if seek_time > 0:
                try:
                    # Try to modify the URL to include the seek time
                    if '?' in stream_url:
                        stream_url += f"&start={seek_time}"
                    else:
                        stream_url += f"?start={seek_time}"
                    logger.info(f"Added seek parameter to URL: start={seek_time}")
                except Exception as e:
                    logger.error(f"Error adding seek parameter: {e}")
            
            # Open the video stream
            cap = cv2.VideoCapture(stream_url)
            
            # Set buffer size to improve streaming
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
            
            if not cap.isOpened():
                logger.error(f"Failed to open direct video stream")
                return self._get_fallback_stream(url)
            
            # Log video properties
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0 or video_fps > 60:
                video_fps = fps  # Use the FPS from metadata
            logger.info(f"Direct stream properties: {width}x{height}, {video_fps} fps")
            
            # Handle seeking for direct streams (less reliable)
            if seek_time > 0 and video_fps > 0:
                frame_pos = int(seek_time * video_fps)
                logger.info(f"Seeking direct stream to frame {frame_pos} (time: {seek_time}s)")
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            
            # Return a capture with frame rate control
            video_id = self._extract_video_id(url) or f"direct_{hash(url)}"
            return self._create_rate_controlled_capture(cap, video_id, video_fps)
            
        except Exception as e:
            logger.error(f"Error setting up direct stream: {e}")
            logger.error(traceback.format_exc())
            return self._get_fallback_stream(url)
    
    def _get_fallback_stream(self, url):
        """Last-resort fallback to ensure a video stream is returned."""
        try:
            logger.warning(f"Using last-resort fallback for URL: {url}")
            
            # Try directly with yt-dlp in the most compatible mode
            ydl_opts = {
                'format': 'worst[ext=mp4]',  # Use worst quality for maximum compatibility
                'quiet': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                stream_url = info.get('url')
                fps = info.get('fps', 30)
                
                if not stream_url:
                    # Create a black frame video as absolute last resort
                    return self._create_dummy_stream()
                
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    return self._create_dummy_stream()
                
                # Return with rate control
                video_id = self._extract_video_id(url) or f"fallback_{hash(url)}"
                return self._create_rate_controlled_capture(cap, video_id, fps)
                
        except Exception:
            return self._create_dummy_stream()
    
    def _create_dummy_stream(self):
        """Create a dummy video stream with a message as absolute last resort."""
        logger.warning("Creating dummy stream with error message")
        # This is a custom VideoCapture-like class that yields error frames
        
        class DummyCapture:
            def __init__(self):
                self.frame_count = 0
                self.is_open = True
                self.last_frame_time = time.time()
                self.fps = 15
                self.frame_time = 1.0 / self.fps
            
            def isOpened(self):
                return self.is_open
            
            def read(self):
                if not self.is_open:
                    return False, None
                
                # Control frame rate
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                if elapsed < self.frame_time:
                    time.sleep(self.frame_time - elapsed)
                
                # Create a black frame with error text
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                
                # Add text that alternates to catch attention
                if (self.frame_count // 30) % 2 == 0:
                    cv2.putText(
                        frame, 
                        "YouTube Stream Error", 
                        (180, 160), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 0, 255), 
                        2
                    )
                    cv2.putText(
                        frame, 
                        "Fallback Mode Active", 
                        (160, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 0, 255), 
                        2
                    )
                
                self.frame_count += 1
                self.last_frame_time = time.time()
                return True, frame
            
            def get(self, prop_id):
                if prop_id == cv2.CAP_PROP_FPS:
                    return self.fps
                elif prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                    return 640
                elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                    return 360
                return 0
            
            def set(self, prop_id, value):
                return True
            
            def release(self):
                self.is_open = False
        
        return DummyCapture()
    
    def _cleanup_old_videos(self):
        """Periodically remove old cached videos to free up space."""
        while not self.stop_cleanup.is_set():
            try:
                # Sleep for the cleanup interval
                for _ in range(self.cleanup_interval):
                    time.sleep(1)
                    if self.stop_cleanup.is_set():
                        return
                
                # Find videos to remove (not accessed in the last 2 hours)
                current_time = time.time()
                videos_to_remove = []
                
                with self.lock:
                    for video_id, info in self.downloads.items():
                        if (info['status'] == 'complete' and 
                            current_time - info['last_accessed'] > 7200):  # 2 hours
                            videos_to_remove.append((video_id, info['path']))
                
                # Remove old videos
                for video_id, path in videos_to_remove:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                            logger.info(f"Removed old cached video: {path}")
                        except Exception as e:
                            logger.error(f"Error removing cached video {path}: {e}")
                    
                    # Remove from downloads dict
                    with self.lock:
                        if video_id in self.downloads:
                            del self.downloads[video_id]
                        # Also clean up frame buffer
                        if video_id in self.frame_buffer:
                            del self.frame_buffer[video_id]
            
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}")
    
    def stop(self):
        """Stop the cleanup thread."""
        self.stop_cleanup.set()
        if self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        
        # Clean up frame buffers
        for video_id in list(self.frame_buffer.keys()):
            try:
                buffer = self.frame_buffer[video_id]
                while not buffer.empty():
                    buffer.get_nowait()
                del self.frame_buffer[video_id]
            except Exception as e:
                logger.error(f"Error cleaning up frame buffer for {video_id}: {e}")

# Global instance to be used by other modules
youtube_downloader = None

def get_youtube_downloader():
    """Get or create the global YouTube downloader instance."""
    global youtube_downloader
    if youtube_downloader is None:
        youtube_downloader = YouTubeDownloader()
    return youtube_downloader

def process_youtube_stream(url, seek_time=0):
    """Process a YouTube video stream with caching for smoother playback."""
    downloader = get_youtube_downloader()
    try:
        logger.info(f"Processing YouTube URL: {url} with seek_time: {seek_time}")
        cap = downloader.get_video_stream(url, seek_time)
        
        if cap is None:
            logger.error(f"Failed to get video stream for URL: {url}")
            raise RuntimeError(f"Failed to open video stream from YouTube URL")
        
        # Verify the capture object works by reading a test frame
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            logger.warning("First frame test failed, recreating capture object")
            # Release the failed capture
            cap.release()
            # Try direct streaming as fallback
            cap = downloader._get_fallback_stream(url)
        else:
            # Reset to start (or seek position)
            if seek_time > 0:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    frame_pos = int(seek_time * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        return cap
    except Exception as e:
        logger.error(f"Error in process_youtube_stream: {e}")
        logger.error(traceback.format_exc())
        # Always return a valid capture object (even if it's a dummy)
        return downloader._get_fallback_stream(url)