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
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3

    def start(self):
        if not self.running.is_set():
            self.running.set()
            self.thread = threading.Thread(target=self._update_frame)
            self.thread.daemon = True
            self.thread.start()

    def _update_frame(self):
        while self.running.is_set() and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                print(f"Starting stream generator for {self.input_value}")
                self.stream_generator = process_stream(
                    self.input_type, 
                    self.input_value, 
                    seek_time=self.seek_time
                )
                
                for frame in self.stream_generator:
                    if not self.running.is_set():
                        print("Stream stopped")
                        break
                        
                    with self.frame_lock:
                        self.frame = frame
                        self.last_frame_time = time.time()
                
                print("Stream generator exhausted")
                break
            except Exception as e:
                print(f"Stream update error: {e}")
                self.reconnect_attempts += 1
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    time.sleep(2)
                    continue
                break

    def get_frame(self):
        with self.frame_lock:
            current_time = time.time()
            if current_time - self.last_frame_time > 5:  # Reduced timeout to 5 seconds
                self.stop()
                return None
            return self.frame

    def stop(self):
        if self.running.is_set():
            self.running.clear()
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2)  # Reduced timeout to 2 seconds
                if self.thread.is_alive():
                    print("Warning: Thread didn't terminate properly")
                    
def cleanup_streams():
    with stream_lock:
        for stream in active_streams.values():
            stream.stop()
        active_streams.clear()

@app.route("/")
def index():
    return render_template("index.html")

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
            print(f"Error adding camera: {e}")
            return render_template("home.html", error="Failed to add camera")

    try:
        cameras_ref = db.collection("cameras")
        cameras = [doc.to_dict() for doc in cameras_ref.stream()]
        return render_template("home.html", cameras=cameras)
    except:
        return render_template("home.html", cameras=[], error="Failed to load cameras")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/faq")
def faq():
    return render_template("faq.html")

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
        
        return render_template("cameras.html", cameras=cameras)
    except Exception as e:
        print(f"Error fetching cameras: {e}")
        return render_template("cameras.html", cameras=[], error=str(e))
    
@app.route('/video_feed/<camera_name>')
def video_feed(camera_name):
    def generate():
        try:
            logger.info(f"Starting video feed for {camera_name}")
            with stream_lock:
                if camera_name not in active_streams:
                    camera_ref = db.collection("cameras").where("camera_name", "==", camera_name).limit(1).get()
                    
                    if not camera_ref:
                        logger.error(f"Camera {camera_name} not found in database")
                        return
                        
                    camera_doc = camera_ref[0]
                    camera = camera_doc.to_dict()
                    
                    # Stop existing stream if any
                    if camera_name in active_streams:
                        active_streams[camera_name].stop()
                        
                    stream = CameraStream(camera["input_type"], camera["input_value"])
                    stream.start()
                    active_streams[camera_name] = stream
                    
                    # Give the stream some time to initialize
                    time.sleep(1)

            while True:
                if camera_name not in active_streams:
                    logger.error(f"Stream {camera_name} not found in active streams")
                    break
                    
                frame = active_streams[camera_name].get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                    
                try:
                    yield frame
                except Exception as e:
                    logger.error(f"Error yielding frame: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Video feed error for {camera_name}: {e}")
            if camera_name in active_streams:
                active_streams[camera_name].stop()
                del active_streams[camera_name]
        
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera/<camera_name>/seek', methods=['POST'])
def seek_video(camera_name):
    try:
        data = request.get_json()
        seconds = data.get('seconds', 0)
        
        with stream_lock:
            if camera_name in active_streams:
                # Stop existing stream
                active_streams[camera_name].stop()
                
                # Get camera details from database
                camera_ref = db.collection("cameras").where("camera_name", "==", camera_name).limit(1).get()
                if not camera_ref:
                    return jsonify({"success": False, "error": "Camera not found"})
                
                camera = camera_ref[0].to_dict()
                
                # Create new stream with seek parameter
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
        # Stop and remove active stream
        with stream_lock:
            if camera_name in active_streams:
                active_streams[camera_name].stop()
                del active_streams[camera_name]
        
        # Remove from Firestore
        camera_ref = db.collection("cameras").where("camera_name", "==", camera_name).limit(1).get()
        if camera_ref:
            camera_ref[0].reference.delete()
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    if not is_running_from_reloader():
        webbrowser.open("http://127.0.0.1:5000")
    try:
        app.run(debug=True, threaded=True, use_reloader=True)
    finally:
        cleanup_streams()