#app.py
import os
import threading
import webbrowser
import time
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from process_stream import process_stream
from werkzeug.serving import is_running_from_reloader
import logging
import cv2
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Firebase Admin SDK Initialization
try:
    cred = credentials.Certificate("firebase_admin_key.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    logger.error(f"Firebase initialization error: {e}")
    db = None

# Store active camera streams with thread-safe dictionary
active_streams = {}
stream_lock = threading.Lock()

class CameraStream:
    def __init__(self, input_type, input_value, seek_time=0):
        self.input_type = input_type
        self.input_value = input_value
        self.seek_time = seek_time
        self.thread = None
        self.frame = None
        self.running = threading.Event()
        self.frame_lock = threading.Lock()
        self.stream_generator = None
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.start_time = time.time()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        self.video_capture = None

    def start(self):
        if not self.running.is_set():
            self.running.set()
            self.thread = threading.Thread(target=self._update_frame)
            self.thread.daemon = True
            self.thread.start()
            logger.info(f"Started camera stream thread for {self.input_value}")

    def _update_frame(self):
        while self.running.is_set() and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                logger.info(f"Starting stream generator for {self.input_value}")
                self.stream_generator = process_stream(
                    self.input_type, 
                    self.input_value, 
                    seek_time=self.seek_time
                )
                
                # Store VideoCapture object if available
                if hasattr(self.stream_generator, 'cap'):
                    self.video_capture = self.stream_generator.cap
                
                for frame in self.stream_generator:
                    if not self.running.is_set():
                        logger.info(f"Stream stopped for {self.input_value}")
                        return  # Exit immediately
                        
                    with self.frame_lock:
                        self.frame = frame
                        self.last_frame_time = time.time()
                        self.frame_count += 1
                        
                        if self.frame_count % 30 == 0:
                            elapsed = time.time() - self.start_time
                            fps = self.frame_count / elapsed
                            logger.info(f"Stream {self.input_value}: {self.frame_count} frames, {fps:.2f} fps")
                
                if not self.running.is_set():
                    return  # Exit if stopped
                    
            except Exception as e:
                logger.error(f"Stream update error for {self.input_value}: {e}")
                self.reconnect_attempts += 1
                if self.reconnect_attempts < self.max_reconnect_attempts and self.running.is_set():
                    logger.info(f"Attempting reconnect {self.reconnect_attempts}/{self.max_reconnect_attempts}")
                    time.sleep(2)
                    continue
                logger.error(f"Max reconnection attempts reached for {self.input_value}")
                break

    def get_frame(self):
        with self.frame_lock:
            if not self.running.is_set():
                return None
                
            current_time = time.time()
            if current_time - self.last_frame_time > 5:
                logger.warning(f"No frames received for 5 seconds from {self.input_value}")
                return None
                
            if self.frame is None:
                return None
                
            try:
                if isinstance(self.frame, bytes) and self.frame.startswith(b'--frame'):
                    return self.frame
                    
                ret, jpeg = cv2.imencode('.jpg', self.frame)
                if not ret:
                    logger.error(f"Failed to encode frame as JPEG for {self.input_value}")
                    return None
                    
                return b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            except Exception as e:
                logger.error(f"Error encoding frame for {self.input_value}: {e}")
                return None

    def force_stop(self):
        """Force stop all stream processing"""
        logger.info(f"Force stopping stream for {self.input_value}")
        
        # Clear running flag first to stop frame processing
        self.running.clear()
        
        # Stop the generator if it exists
        if self.stream_generator:
            try:
                self.stream_generator.close()
            except Exception as e:
                logger.error(f"Error closing generator: {e}")
            self.stream_generator = None
        
        # Release VideoCapture if it exists
        if self.video_capture is not None:
            try:
                self.video_capture.release()
            except Exception as e:
                logger.error(f"Error releasing video capture: {e}")
            self.video_capture = None
        
        # Stop the thread
        if self.thread and self.thread.is_alive():
            try:
                self.thread.join(timeout=3)
                if self.thread.is_alive():
                    logger.warning(f"Thread didn't terminate for {self.input_value}")
            except Exception as e:
                logger.error(f"Error joining thread: {e}")
        
        # Clear frame
        with self.frame_lock:
            self.frame = None
        
        logger.info(f"Force stop completed for {self.input_value}")

    def stop(self):
        """Stop method that calls force_stop"""
        self.force_stop()

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.force_stop()

def get_connected_cameras():
    """Test and get all available cameras with detailed information"""
    camera_info = []
    max_cameras_to_check = 10

    for i in range(max_cameras_to_check):
        try:
            logger.info(f"Testing camera index {i}")
            # Try different backends
            for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
                cap = cv2.VideoCapture(i + backend)
                
                if cap.isOpened():
                    # Get camera properties
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    backend_name = cap.getBackendName()
                    
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret:
                        camera_info.append({
                            "index": i,
                            "name": f"Camera {i} ({width}x{height})",
                            "width": width,
                            "height": height,
                            "fps": fps,
                            "backend": backend_name
                        })
                        logger.info(f"Successfully detected camera {i}")
                        break  # Break if camera is successfully detected
                    
                cap.release()
                
        except Exception as e:
            logger.error(f"Error testing camera {i}: {str(e)}")
    
    return camera_info

@app.route("/")
def index():
    return render_template("index.html", current_page='index')

@app.route("/debug_cameras")
def debug_cameras():
    """Debug endpoint to check camera detection"""
    cameras = get_connected_cameras()
    return jsonify({
        "cameras_found": len(cameras),
        "camera_details": cameras
    })

@app.route("/get_cameras")
def get_cameras_endpoint():
    """API endpoint to get list of connected cameras"""
    try:
        cameras = get_connected_cameras()
        return jsonify(cameras)
    except Exception as e:
        logger.error(f"Error getting cameras: {e}")
        return jsonify([]), 500

@app.route("/reset_password", methods=["POST"])
def reset_password():
    if db is None:
        return jsonify({"error": "Firebase not initialized"}), 500

    try:
        email = request.json.get("email")
        if not email:
            return jsonify({"error": "Email is required"}), 400
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error in password reset: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/home", methods=["GET", "POST"])
def home():
    if db is None:
        return "Firebase not initialized", 500

    if request.method == "POST":
        try:
            input_type = request.form["input_type"]
            camera_name = request.form["camera_name"]
            input_value = request.form["input_value"]
            google_maps_link = request.form.get("google_maps_link", "")
            mobile_number = request.form.get("mobile_number", "")

            camera_data = {
                "input_type": input_type,
                "camera_name": camera_name,
                "input_value": input_value,
                "google_maps_link": google_maps_link,
                "mobile_number": mobile_number,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            db.collection("cameras").add(camera_data)

            with stream_lock:
                if camera_name not in active_streams:
                    stream = CameraStream(input_type, input_value)
                    stream.start()
                    active_streams[camera_name] = stream

            return redirect(url_for("cameras"))
        except Exception as e:
            logger.error(f"Error adding camera: {e}")
            return render_template("home.html", error="Failed to add camera", current_page='home')

    try:
        cameras_ref = db.collection("cameras")
        cameras = [doc.to_dict() for doc in cameras_ref.stream()]
        return render_template("home.html", cameras=cameras, current_page='home')
    except Exception as e:
        logger.error(f"Error loading cameras: {e}")
        return render_template("home.html", cameras=[], error="Failed to load cameras", current_page='home')
    
@app.route("/detection-history")
def detection_history():
    if db is None:
        return render_template("detection_history.html", error="Firebase not initialized"), 500

    try:
        logs_ref = db.collection("detection_logs").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(50)
        logs = [doc.to_dict() for doc in logs_ref.stream()]
        return render_template("detection_history.html", logs=logs)
    except Exception as e:
        return render_template("detection_history.html", error=str(e))

@app.route("/warnings")
def warnings():
    if db is None:
        return render_template("warnings.html", error="Firebase not initialized"), 500

    try:
        warnings_ref = db.collection("warnings").where("active", "==", True).order_by("timestamp", direction=firestore.Query.DESCENDING)
        warnings = [doc.to_dict() for doc in warnings_ref.stream()]
        return render_template("warnings.html", warnings=warnings)
    except Exception as e:
        return render_template("warnings.html", error=str(e))
    
@app.route("/about")
def about():
    return render_template("about.html", current_page='about')

@app.route("/contact")
def contact():
    return render_template("contact.html", current_page='contact')

@app.route("/faq")
def faq():
    return render_template("faq.html", current_page='faq')

@app.route("/cameras")
def cameras():
    if db is None:
        return "Firebase not initialized", 500

    try:
        cameras_ref = db.collection("cameras").stream()
        cameras = []
        for doc in cameras_ref:
            camera_data = doc.to_dict()
            camera_data['id'] = doc.id
            cameras.append(camera_data)
        
        return render_template("cameras.html", cameras=cameras, current_page='cameras')
    except Exception as e:
        logger.error(f"Error fetching cameras: {e}")
        return render_template("cameras.html", cameras=[], error=str(e), current_page='cameras')

@app.route('/video_feed/<camera_name>')
def video_feed(camera_name):
    def generate():
        try:
            logger.info(f"Starting video feed for {camera_name}")
            with stream_lock:
                if camera_name not in active_streams:
                    logger.info(f"Creating new stream for {camera_name}")
                    camera_ref = db.collection("cameras").where("camera_name", "==", camera_name).limit(1).get()
                    
                    if not camera_ref:
                        logger.error(f"Camera {camera_name} not found in database")
                        return
                    
                    camera_doc = list(camera_ref)[0]
                    camera = camera_doc.to_dict()
                    logger.info(f"Found camera config: {camera}")
                    
                    # Stop existing stream if any
                    if camera_name in active_streams:
                        logger.info(f"Stopping existing stream for {camera_name}")
                        active_streams[camera_name].force_stop()  # Use force_stop instead of stop
                        del active_streams[camera_name]
                    
                    # Create new stream
                    logger.info(f"Creating stream with type: {camera['input_type']}, value: {camera['input_value']}")
                    stream = CameraStream(camera["input_type"], camera["input_value"])
                    stream.start()
                    active_streams[camera_name] = stream
                    logger.info(f"Stream started for {camera_name}")

            frame_count = 0
            start_time = time.time()
            last_frame_time = time.time()
            
            while True:
                # Check if camera still exists in database
                camera_exists = False
                try:
                    camera_ref = db.collection("cameras").where("camera_name", "==", camera_name).limit(1).get()
                    camera_exists = len(list(camera_ref)) > 0
                except:
                    pass

                if not camera_exists:
                    logger.info(f"Camera {camera_name} no longer exists in database, stopping feed")
                    break

                if camera_name not in active_streams:
                    logger.error(f"Stream {camera_name} not found in active streams")
                    break
                
                frame = active_streams[camera_name].get_frame()
                current_time = time.time()
                
                if frame is None:
                    if current_time - last_frame_time > 5:  # No frames for 5 seconds
                        logger.warning(f"No frames received for {camera_name}")
                        break
                    time.sleep(0.1)
                    continue
                
                last_frame_time = current_time
                frame_count += 1
                
                try:
                    yield frame
                except Exception as e:
                    logger.error(f"Error yielding frame for {camera_name}: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Video feed error for {camera_name}: {e}")
        finally:
            # Cleanup in finally block
            try:
                if camera_name in active_streams:
                    active_streams[camera_name].force_stop()
                    del active_streams[camera_name]
                    gc.collect()
            except Exception as cleanup_error:
                logger.error(f"Error in final cleanup: {cleanup_error}")
            logger.info(f"Stream ended for {camera_name}")
    
    response = Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response
    
@app.route('/camera/<camera_name>/details')
def camera_details(camera_name):
    try:
        camera_ref = db.collection("cameras").where("camera_name", "==", camera_name).limit(1).get()
        if not camera_ref:
            return jsonify({"error": "Camera not found"}), 404
            
        camera_data = camera_ref[0].to_dict()
        return jsonify(camera_data)
    except Exception as e:
        logger.error(f"Error getting camera details: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/camera/<camera_name>/seek', methods=['POST'])
def seek_video(camera_name):
    try:
        data = request.get_json()
        seconds = data.get('seconds', 0)
        
        with stream_lock:
            if camera_name in active_streams:
                active_streams[camera_name].stop()
                
                camera_ref = db.collection("cameras").where("camera_name", "==", camera_name).limit(1).get()
                if not camera_ref:
                    return jsonify({"success": False, "error": "Camera not found"})
                
                camera = camera_ref[0].to_dict()
                
                stream = CameraStream(camera["input_type"], camera["input_value"], seek_time=seconds)
                stream.start()
                active_streams[camera_name] = stream
                
                return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error seeking video: {e}")
        return jsonify({"success": False, "error": str(e)})

def cleanup_streams():
    """Clean up all active camera streams when the application shuts down"""
    logger.info("Starting global cleanup of all camera streams")
    try:
        with stream_lock:
            camera_names = list(active_streams.keys())
            
            for camera_name in camera_names:
                try:
                    logger.info(f"Force cleaning up stream for camera: {camera_name}")
                    cleanup_camera_stream(camera_name)
                except Exception as e:
                    logger.error(f"Error cleaning up stream for {camera_name}: {e}")
            
            # Clear all streams
            active_streams.clear()
            
            # Force Python garbage collection
            import gc
            gc.collect()
            
        logger.info("Global cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during global cleanup: {e}")
        # Emergency cleanup
        try:
            active_streams.clear()
            gc.collect()
        except:
            pass

def cleanup_camera_stream(camera_name):
    """Helper function to clean up camera stream resources"""
    try:
        if camera_name in active_streams:
            logger.info(f"Starting force cleanup for camera: {camera_name}")
            stream = active_streams[camera_name]
            
            # Force stop all processing
            stream.force_stop()
            
            # Stop the generator from process_stream module
            if hasattr(stream, 'stream_generator'):
                try:
                    # Try to access and cleanup VideoProcessor
                    if hasattr(stream.stream_generator, 'video_processor'):
                        try:
                            del stream.stream_generator.video_processor.models
                            del stream.stream_generator.video_processor
                        except:
                            pass
                except:
                    pass
            
            # Remove from active streams
            del active_streams[camera_name]
            
            # Force Python garbage collection multiple times
            import gc
            for _ in range(3):
                gc.collect()
            
            logger.info(f"Successfully cleaned up stream for camera: {camera_name}")
            return True
    except Exception as e:
        logger.error(f"Error during stream cleanup for {camera_name}: {e}")
        return False

@app.route('/camera/<camera_name>/remove', methods=['POST'])
def remove_camera(camera_name):
    if db is None:
        return jsonify({"error": "Firebase not initialized"}), 500

    logger.info(f"Starting camera removal process for: {camera_name}")

    try:
        # 1. Stop the video feed first to prevent new connections
        with stream_lock:
            if camera_name in active_streams:
                logger.info(f"Stopping stream for camera: {camera_name}")
                stream = active_streams[camera_name]
                # Force stop all processing
                stream.force_stop()
                # Remove from active streams
                del active_streams[camera_name]

        # 2. Then remove from database
        camera_ref = db.collection("cameras").where("camera_name", "==", camera_name).limit(1).get()
        camera_doc = None
        
        for doc in camera_ref:
            camera_doc = doc
            break
            
        if camera_doc:
            camera_doc.reference.delete()
            logger.info(f"Deleted camera {camera_name} from database")
        else:
            logger.warning(f"Camera {camera_name} not found in database")

        # 3. Force garbage collection multiple times
        import gc
        for _ in range(3):
            gc.collect()
        
        logger.info(f"Successfully removed camera {camera_name}")
        
        # 4. Add a small delay to ensure cleanup is complete
        time.sleep(0.5)
        
        return jsonify({"success": True})

    except Exception as e:
        logger.error(f"Error removing camera {camera_name}: {e}")
        # Emergency cleanup
        try:
            if camera_name in active_streams:
                stream = active_streams[camera_name]
                stream.force_stop()
                del active_streams[camera_name]
                gc.collect()
        except Exception as cleanup_error:
            logger.error(f"Emergency cleanup failed: {cleanup_error}")
        
        return jsonify({"success": False, "error": str(e)}), 500
if __name__ == "__main__":
    if not is_running_from_reloader():
        webbrowser.open("http://127.0.0.1:5000")
    try:
        app.run(debug=True, threaded=True, use_reloader=True)
    finally:
        cleanup_streams()