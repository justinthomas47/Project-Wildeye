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
                
                for frame in self.stream_generator:
                    if not self.running.is_set():
                        logger.info(f"Stream stopped for {self.input_value}")
                        break
                        
                    with self.frame_lock:
                        self.frame = frame
                        self.last_frame_time = time.time()
                        self.frame_count += 1
                        
                        if self.frame_count % 30 == 0:  # Log every 30 frames
                            elapsed = time.time() - self.start_time
                            fps = self.frame_count / elapsed
                            logger.info(f"Stream {self.input_value}: {self.frame_count} frames, {fps:.2f} fps")
                
                logger.info(f"Stream generator exhausted for {self.input_value}")
                break
            except Exception as e:
                logger.error(f"Stream update error for {self.input_value}: {e}")
                self.reconnect_attempts += 1
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    logger.info(f"Attempting reconnect {self.reconnect_attempts}/{self.max_reconnect_attempts}")
                    time.sleep(2)
                    continue
                logger.error(f"Max reconnection attempts reached for {self.input_value}")
                break

    def get_frame(self):
        with self.frame_lock:
            current_time = time.time()
            if current_time - self.last_frame_time > 5:
                logger.warning(f"No frames received for 5 seconds from {self.input_value}")
                return None
                
            if self.frame is None:
                return None
                
            try:
                # Check if frame is already encoded
                if isinstance(self.frame, bytes) and self.frame.startswith(b'--frame'):
                    return self.frame
                    
                # Encode frame as JPEG if it's a numpy array
                ret, jpeg = cv2.imencode('.jpg', self.frame)
                if not ret:
                    logger.error(f"Failed to encode frame as JPEG for {self.input_value}")
                    return None
                    
                return b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            except Exception as e:
                logger.error(f"Error encoding frame for {self.input_value}: {e}")
                return None

    def stop(self):
        if self.running.is_set():
            logger.info(f"Stopping stream for {self.input_value}")
            self.running.clear()
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2)
                if self.thread.is_alive():
                    logger.warning(f"Thread didn't terminate properly for {self.input_value}")
            logger.info(f"Stream stopped for {self.input_value}")

def cleanup_streams():
    with stream_lock:
        for stream in active_streams.values():
            stream.stop()
        active_streams.clear()

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
                    
                    camera_doc = list(camera_ref)[0]  # Convert to list to access first item
                    camera = camera_doc.to_dict()
                    logger.info(f"Found camera config: {camera}")
                    
                    # Stop existing stream if any
                    if camera_name in active_streams:
                        logger.info(f"Stopping existing stream for {camera_name}")
                        active_streams[camera_name].stop()
                    
                    # Create new stream
                    logger.info(f"Creating stream with type: {camera['input_type']}, value: {camera['input_value']}")
                    stream = CameraStream(camera["input_type"], camera["input_value"])
                    stream.start()
                    active_streams[camera_name] = stream
                    logger.info(f"Stream started for {camera_name}")
                else:
                    logger.info(f"Using existing stream for {camera_name}")

            frame_count = 0
            start_time = time.time()
            
            while True:
                if camera_name not in active_streams:
                    logger.error(f"Stream {camera_name} not found in active streams")
                    break
                
                frame = active_streams[camera_name].get_frame()
                
                if frame is None:
                    logger.warning(f"No frame received for {camera_name}")
                    if time.time() - start_time > 30:  # After 30 seconds of no frames
                        logger.error(f"No frames received for 30 seconds, stopping stream for {camera_name}")
                        break
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                if frame_count % 30 == 0:  # Log every 30 frames
                    fps = frame_count / (time.time() - start_time)
                    logger.info(f"Camera {camera_name}: Processed {frame_count} frames, {fps:.2f} fps")
                
                try:
                    yield frame
                except Exception as e:
                    logger.error(f"Error yielding frame for {camera_name}: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Video feed error for {camera_name}: {e}")
            if camera_name in active_streams:
                active_streams[camera_name].stop()
                del active_streams[camera_name]
        finally:
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

@app.route('/camera/<camera_name>/remove', methods=['POST'])
def remove_camera(camera_name):
    if db is None:
        return jsonify({"error": "Firebase not initialized"}), 500

    try:
        with stream_lock:
            if camera_name in active_streams:
                active_streams[camera_name].stop()
                del active_streams[camera_name]
        
        camera_ref = db.collection("cameras").where("camera_name", "==", camera_name).limit(1).get()
        if camera_ref:
            camera_ref[0].reference.delete()
        
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error removing camera: {e}")
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    if not is_running_from_reloader():
        webbrowser.open("http://127.0.0.1:5000")
    try:
        app.run(debug=True, threaded=True, use_reloader=True)
    finally:
        cleanup_streams()