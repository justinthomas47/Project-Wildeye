#app.py
from logging_config import configure_logging
configure_logging()
from dotenv import load_dotenv
load_dotenv()
import os
import threading
import webbrowser
import time
import uuid
import re
import secrets
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, session
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore, auth
from process_stream import process_stream, pause_stream, resume_stream
from werkzeug.serving import is_running_from_reloader
from detection_handler import init_drive_service as init_detection_drive
import logging
import cv2
import gc
import numpy as np
import traceback
from warning_system import (
    get_all_warnings, 
    acknowledge_warning, 
    resolve_warning, 
    get_notification_preferences,
    update_notification_preferences,
    test_notification_channels
)
from functools import wraps

# Set up logger for this module
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure session with a secure secret key
app.secret_key = os.environ.get('SECRET_KEY') or secrets.token_hex(16)
# Session settings
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # Session lifetime in seconds (1 hour)
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Firebase Admin SDK Initialization
try:
    cred = credentials.Certificate("firebase_admin_key.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    logger.error(f"Firebase initialization error: {e}")
    db = None

# Add this after Firebase initialization
try:
    # Initialize detection handler
    init_detection_drive()
    logger.info("Detection handler initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize detection handler: {e}")

# Store active camera streams with thread-safe dictionary
active_streams = {}
stream_lock = threading.Lock()

# Authentication helper functions
def get_current_user():
    """Get the currently authenticated user from session"""
    try:
        # Check if session contains user info
        if 'user_id' in session:
            return {
                'uid': session['user_id'],
                'email': session.get('user_email', ''),
                'name': session.get('user_name', '')
            }
        return None
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        return None

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if user is None:
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

class CameraStream:
    def __init__(self, input_type, input_value, seek_time=0, visible=False, camera_id=None):
        self.input_type = input_type
        self.input_value = input_value
        self.seek_time = seek_time
        self.thread = None
        self.frame = None
        self.running = threading.Event()
        self.is_visible = visible  # Track if stream is visible in UI
        self.frame_lock = threading.Lock()
        self.stream_generator = None
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.start_time = time.time()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        self.video_capture = None
        self.camera_id = camera_id  # Store the unique camera ID
        self.static_frame = self._create_static_frame()
        
        # Log initialization
        logger.info(f"Initialized CameraStream: type={input_type}, value={input_value}, visible={visible}, camera_id={camera_id}")

    def _create_static_frame(self):
        """Create a static frame for when the stream is paused"""
        try:
            # Create a black frame with text
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame, 
                "Stream inactive - Click to activate", 
                (120, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            
            # Convert to JPEG bytes
            _, buffer = cv2.imencode('.jpg', frame)
            return b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        except Exception as e:
            logger.error(f"Error creating static frame: {e}")
            return None

    def set_visibility(self, visible):
        """Update the visibility status of the stream"""
        if self.is_visible == visible:
            return  # No change needed
            
        logger.info(f"Setting visibility for {self.input_value} to {visible}")
        self.is_visible = visible
        
        # Pause or resume the backend processing based on visibility
        if visible:
            resume_stream(self.input_value)
        else:
            pause_stream(self.input_value)

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
                logger.info(f"Starting stream generator for {self.input_value} (visible={self.is_visible})")
                self.stream_generator = process_stream(
                    self.input_type, 
                    self.input_value, 
                    seek_time=self.seek_time,
                    camera_id=self.camera_id,  # Pass camera_id instead of input_value
                    db=db,
                    is_visible=self.is_visible
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
        """
        Get the current frame from the camera stream.
        Returns the frame as JPEG bytes with multipart/x-mixed-replace content type headers.
        """
        # If not visible and we have a static frame, return that instead
        if not self.is_visible and self.static_frame is not None:
            return self.static_frame
            
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
                # If frame is already in the correct format, return it
                if isinstance(self.frame, bytes) and self.frame.startswith(b'--frame'):
                    return self.frame
                    
                # Otherwise, encode the frame as JPEG
                ret, jpeg = cv2.imencode('.jpg', self.frame)
                if not ret:
                    logger.error(f"Failed to encode frame as JPEG for {self.input_value}")
                    return None
                    
                # Return with proper multipart headers
                frame_data = b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
                return frame_data
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

def create_error_frame(error_message):
    """Create an error frame with text"""
    try:
        # Create a black frame with error text
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Add dark red background
        frame[:, :] = (0, 0, 100)  # Dark red in BGR
        
        # Split message into multiple lines if it's long
        max_width = 40  # characters per line
        lines = []
        for i in range(0, len(error_message), max_width):
            lines.append(error_message[i:i+max_width])
        
        y_position = 100
        for line in lines:
            cv2.putText(
                frame, 
                line, 
                (20, y_position), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255),  # White text
                1
            )
            y_position += 25
        
        # Convert to JPEG bytes
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()
    except Exception as e:
        logger.error(f"Error creating error frame: {e}")
        # Return a minimal error image
        return b''  # Empty byte string as fallback

@app.route("/")
def index():
    return render_template("index.html", current_page='index')

@app.route('/set-session', methods=['POST'])
def set_session():
    """Set user session data without token verification"""
    try:
        data = request.json
        if not data or 'uid' not in data:
            return jsonify({'error': 'Invalid data'}), 400
            
        # Set session variables
        session['user_id'] = data['uid']
        session['user_email'] = data.get('email', '')
        session['user_name'] = data.get('name', '')
        
        # Check if user exists in database, create if not
        user_ref = db.collection('users').document(data['uid'])
        if not user_ref.get().exists:
            user_data = {
                'uid': data['uid'],
                'email': data.get('email', ''),
                'name': data.get('name', ''),
                'created_at': datetime.now()
            }
            user_ref.set(user_data)
        
        # Return success response
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Session error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/logout-backend', methods=['POST'])
def logout_backend():
    """Clear the user session"""
    session.clear()
    return jsonify({'success': True})

@app.route("/debug_cameras")
@login_required
def debug_cameras():
    """Debug endpoint to check camera detection"""
    cameras = get_connected_cameras()
    return jsonify({
        "cameras_found": len(cameras),
        "camera_details": cameras
    })

@app.route('/debug-streams')
@login_required
def debug_streams():
    """Debug endpoint to check active streams and their status"""
    try:
        # Check if user is authenticated
        user = get_current_user()
        if not user:
            return jsonify({"error": "Not authenticated"}), 401
            
        stream_info = []
        
        with stream_lock:
            for camera_name, stream in active_streams.items():
                camera_ref = db.collection("cameras").where("camera_name", "==", camera_name).limit(1).get()
                camera_owner = None
                
                for doc in camera_ref:
                    camera_owner = doc.to_dict().get('owner_uid')
                
                # Only show streams owned by the current user
                if camera_owner == user['uid']:
                    stream_info.append({
                        "camera_name": camera_name,
                        "input_type": stream.input_type,
                        "input_value": stream.input_value,
                        "camera_id": stream.camera_id,
                        "is_visible": stream.is_visible,
                        "is_running": stream.running.is_set(),
                        "reconnect_attempts": stream.reconnect_attempts,
                        "last_frame_time": time.time() - stream.last_frame_time,
                        "frame_count": stream.frame_count
                    })
        
        return jsonify({
            "active_streams": len(stream_info),
            "streams": stream_info
        })
    except Exception as e:
        logger.error(f"Error in debug-streams: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/get_cameras")
@login_required
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
@login_required
def home():
    if db is None:
        return "Firebase not initialized", 500

    user = get_current_user()
    if not user:
        return redirect(url_for('index'))

    if request.method == "POST":
        try:
            input_type = request.form["input_type"]
            camera_name = request.form["camera_name"]
            input_value = request.form["input_value"]
            google_maps_link = request.form.get("google_maps_link", "")
            mobile_number = request.form.get("mobile_number", "")

            # Generate a unique ID for the camera
            clean_name = re.sub(r'[^a-zA-Z0-9]', '', camera_name.lower())
            camera_id = f"{clean_name}_{str(uuid.uuid4())[:8]}"

            camera_data = {
                "camera_id": camera_id,
                "input_type": input_type,
                "camera_name": camera_name,
                "input_value": input_value,
                "google_maps_link": google_maps_link,
                "mobile_number": mobile_number,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "owner_uid": user['uid']  # Add the owner_uid to link data to user
            }
            
            # Save to Firestore with the generated ID as document ID
            db.collection("cameras").document(camera_id).set(camera_data)

            with stream_lock:
                if camera_name not in active_streams:
                    stream = CameraStream(input_type, input_value, camera_id=camera_id, visible=False)
                    stream.start()
                    active_streams[camera_name] = stream

            return redirect(url_for("cameras"))
        except Exception as e:
            logger.error(f"Error adding camera: {e}")
            return render_template("home.html", error="Failed to add camera", current_page='home')

    try:
        # Only get cameras for current user
        cameras_ref = db.collection("cameras").where("owner_uid", "==", user['uid'])
        cameras = [doc.to_dict() for doc in cameras_ref.stream()]
        return render_template("home.html", cameras=cameras, current_page='home')
    except Exception as e:
        logger.error(f"Error loading cameras: {e}")
        return render_template("home.html", cameras=[], error="Failed to load cameras", current_page='home')
    
@app.route("/detection-history")
@login_required
def detection_history():
    if db is None:
        logger.error("Firebase not initialized for detection_history")
        return render_template("detection_history.html", error="Firebase not initialized"), 500

    user = get_current_user()
    if not user:
        return redirect(url_for('index'))

    try:
        # First get all cameras owned by this user
        user_cameras = []
        cameras_ref = db.collection("cameras").where("owner_uid", "==", user['uid']).stream()
        for doc in cameras_ref:
            camera_data = doc.to_dict()
            user_cameras.append(camera_data['camera_name'])
        
        # Then query detection logs, filtering for only the user's cameras
        logs_ref = db.collection("detection_logs").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(50)
        logs = []
        
        for doc in logs_ref.stream():
            log_data = doc.to_dict()
            
            # Only include if camera belongs to user
            if log_data.get('camera') in user_cameras:
                # Format the timestamp for display
                if 'timestamp' in log_data and isinstance(log_data['timestamp'], datetime):
                    log_data['timestamp'] = log_data['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                    
                # Ensure confidence is a number
                if 'confidence' in log_data:
                    if isinstance(log_data['confidence'], str):
                        try:
                            log_data['confidence'] = float(log_data['confidence'])
                        except ValueError:
                            log_data['confidence'] = 95.0  # Default if parsing fails
                else:
                    log_data['confidence'] = 95.0  # Default confidence if not present
                    
                logs.append(log_data)
            
        return render_template("detection_history.html", logs=logs, current_page='detection_history')
    except Exception as e:
        logger.error(f"Error fetching detection logs: {e}")
        logger.error(traceback.format_exc())
        return render_template("detection_history.html", 
                              error=f"Failed to load detection history: {str(e)}", 
                              current_page='detection_history')

@app.route("/warnings")
@login_required
def warnings():
    """Display warning history page with active and resolved warnings"""
    if db is None:
        return render_template("warnings.html", error="Firebase not initialized"), 500

    user = get_current_user()
    if not user:
        return redirect(url_for('index'))

    try:
        # Get all cameras owned by this user
        user_cameras = []
        cameras_ref = db.collection("cameras").where("owner_uid", "==", user['uid']).stream()
        for doc in cameras_ref:
            camera_data = doc.to_dict()
            user_cameras.append(camera_data['camera_id'])
        
        # Get all warnings
        warnings_ref = db.collection('warnings').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(100)
        all_warnings = []
        
        for doc in warnings_ref.stream():
            warning_data = doc.to_dict()
            # Only include warnings for cameras owned by this user
            if warning_data.get('camera_id') in user_cameras:
                all_warnings.append(warning_data)
        
        return render_template("warnings.html", warnings=all_warnings, current_page='warnings')
    except Exception as e:
        logger.error(f"Error loading warnings: {e}")
        logger.error(traceback.format_exc())
        return render_template("warnings.html", error=str(e), current_page='warnings')

@app.route("/warning/<warning_id>/acknowledge", methods=["POST"])
@login_required
def acknowledge_warning_route(warning_id):
    """API endpoint to acknowledge a warning"""
    if db is None:
        return jsonify({"success": False, "error": "Firebase not initialized"}), 500
        
    try:
        success = acknowledge_warning(db, warning_id)
        return jsonify({"success": success})
    except Exception as e:
        logger.error(f"Error acknowledging warning: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/warning/<warning_id>/resolve", methods=["POST"])
@login_required
def resolve_warning_route(warning_id):
    """API endpoint to resolve a warning"""
    if db is None:
        return jsonify({"success": False, "error": "Firebase not initialized"}), 500
        
    try:
        success = resolve_warning(db, warning_id)
        return jsonify({"success": success})
    except Exception as e:
        logger.error(f"Error resolving warning: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/notification_settings", methods=["GET", "POST"])
@login_required
def notification_settings():
    """Page to manage notification settings"""
    if db is None:
        return render_template("notification_settings.html", error="Firebase not initialized"), 500
        
    user = get_current_user()
    if not user:
        return redirect(url_for('index'))
        
    try:
        # Import required functions from sms_service
        from sms_service import get_available_carriers, save_carrier_preference

        # Get available carriers for the dropdown
        carriers = get_available_carriers()
        
        if request.method == "POST":
            # Process form submission
            preferences = {
                'email': {
                    'enabled': 'email_enabled' in request.form,
                    'recipient': request.form.get('email_recipient', '')
                },
                'sms': {
                    'enabled': 'sms_enabled' in request.form,
                    'recipient': request.form.get('sms_recipient', ''),
                    'carrier': request.form.get('sms_carrier', 'default')
                },
                'telegram': {
                    'enabled': 'telegram_enabled' in request.form,
                    'chat_id': request.form.get('telegram_chat_id', '')
                },
                'owner_uid': user['uid']  # Add user ID to preferences
            }
            
            # Save preferences to user-specific document
            doc_ref = db.collection('settings').document(f"notifications_{user['uid']}")
            doc_ref.set(preferences)
            
            # Also save carrier preference for SMS service
            if 'sms_enabled' in request.form and request.form.get('sms_carrier'):
                save_carrier_preference(db, user['uid'], request.form.get('sms_carrier'))
            
            return render_template(
                "notification_settings.html", 
                preferences=preferences, 
                carriers=carriers,
                success="Notification settings updated successfully",
                current_page='settings'
            )
        
        # GET request - show current settings
        doc_ref = db.collection('settings').document(f"notifications_{user['uid']}")
        doc = doc_ref.get()
        
        if doc.exists:
            preferences = doc.to_dict()
        else:
            # Default preferences
            preferences = {
                'email': {
                    'enabled': False,
                    'recipient': user.get('email', '')
                },
                'sms': {
                    'enabled': False,
                    'recipient': '',
                    'carrier': 'default'
                },
                'telegram': {
                    'enabled': False,
                    'chat_id': ''
                },
                'owner_uid': user['uid']
            }
            # Create default settings
            doc_ref.set(preferences)
            
        return render_template(
            "notification_settings.html", 
            preferences=preferences,
            carriers=carriers,
            current_page='settings'
        )
    except Exception as e:
        logger.error(f"Error with notification settings: {e}")
        return render_template(
            "notification_settings.html", 
            error=f"Error: {str(e)}",
            carriers=get_available_carriers() if 'get_available_carriers' in locals() else [],
            current_page='settings'
        )

@app.route("/test_notifications", methods=["POST"])
@login_required
def test_notifications():
    """API endpoint to test notification channels"""
    if db is None:
        return jsonify({"success": False, "error": "Firebase not initialized"}), 500
        
    try:
        # Get current user
        user = get_current_user()
        if not user:
            return jsonify({"success": False, "error": "User not authenticated"}), 401
            
        # Import test function from warning system
        from warning_system import test_notification_channels
        
        # Run test
        test_results = test_notification_channels(db, user['uid'])
        
        return jsonify(test_results)
    except Exception as e:
        logger.error(f"Error testing notifications: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    
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
@login_required
def cameras():
    if db is None:
        return "Firebase not initialized", 500

    user = get_current_user()
    if not user:
        return redirect(url_for('index'))

    try:
        # Only get cameras owned by current user
        cameras_ref = db.collection("cameras").where("owner_uid", "==", user['uid']).stream()
        cameras = []
        for doc in cameras_ref:
            camera_data = doc.to_dict()
            camera_data['id'] = doc.id
            cameras.append(camera_data)
        
        return render_template("cameras.html", cameras=cameras, current_page='cameras')
    except Exception as e:
        logger.error(f"Error fetching cameras: {e}")
        return render_template("cameras.html", cameras=[], error=str(e), current_page='cameras')

@app.route('/camera/<camera_name>/pause', methods=['POST'])
@login_required
def pause_camera_stream(camera_name):
    """Pause processing for a camera when not visible in the UI"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({"success": False, "error": "Not authenticated"}), 401
            
        # Verify camera belongs to user
        camera_docs = list(db.collection("cameras")
                        .where("camera_name", "==", camera_name)
                        .where("owner_uid", "==", user['uid'])
                        .limit(1)
                        .stream())
        
        if not camera_docs:
            return jsonify({"success": False, "error": "Camera not found or access denied"}), 403
            
        with stream_lock:
            if camera_name in active_streams:
                logger.info(f"Pausing stream for camera: {camera_name}")
                active_streams[camera_name].set_visibility(False)
                return jsonify({"success": True})
            else:
                logger.warning(f"Attempted to pause non-existent stream: {camera_name}")
                return jsonify({"success": False, "error": "Stream not found"}), 404
    except Exception as e:
        logger.error(f"Error pausing camera {camera_name}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/camera/<camera_name>/resume', methods=['POST'])
@login_required
def resume_camera_stream(camera_name):
    """Resume processing for a camera when visible in the UI"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({"success": False, "error": "Not authenticated"}), 401
            
        # Verify camera belongs to user
        camera_docs = list(db.collection("cameras")
                        .where("camera_name", "==", camera_name)
                        .where("owner_uid", "==", user['uid'])
                        .limit(1)
                        .stream())
        
        if not camera_docs:
            return jsonify({"success": False, "error": "Camera not found or access denied"}), 403
            
        with stream_lock:
            if camera_name in active_streams:
                logger.info(f"Resuming stream for camera: {camera_name}")
                active_streams[camera_name].set_visibility(True)
                return jsonify({"success": True})
            else:
                logger.warning(f"Attempted to resume non-existent stream: {camera_name}")
                return jsonify({"success": False, "error": "Stream not found"}), 404
    except Exception as e:
        logger.error(f"Error resuming camera {camera_name}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/video_feed/<camera_name>')
@login_required
def video_feed(camera_name):
    # Check if the user is authenticated
    user = get_current_user()
    if not user:
        # Return a static error image if not authenticated
        return Response('Not authenticated', status=403, mimetype='text/plain')
    
    # Check if the stream should be visible (parameter sent by frontend)
    is_visible = request.args.get('visible', 'false').lower() == 'true'
    
    # Log the request for debugging
    logger.info(f"Video feed request for {camera_name} (visible={is_visible}, user={user['uid']})")
    
    def generate():
        try:
            logger.info(f"Starting video feed for {camera_name} (visible={is_visible})")
            with stream_lock:
                if camera_name not in active_streams:
                    logger.info(f"Creating new stream for {camera_name}")
                    
                    # Get camera from database, make sure it belongs to current user
                    camera_docs = list(db.collection("cameras")
                                     .where("camera_name", "==", camera_name)
                                     .where("owner_uid", "==", user['uid'])
                                     .limit(1)
                                     .stream())
                    
                    if not camera_docs:
                        logger.error(f"Camera {camera_name} not found or not owned by current user")
                        yield (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n' + 
                              create_error_frame(f"Camera not found or access denied") + 
                              b'\r\n')
                        return
                    
                    camera_doc = camera_docs[0]
                    camera = camera_doc.to_dict()
                    logger.info(f"Found camera config: {camera}")
                    
                    # Stop existing stream if any
                    if camera_name in active_streams:
                        logger.info(f"Stopping existing stream for {camera_name}")
                        active_streams[camera_name].force_stop()
                        del active_streams[camera_name]
                    
                    # Get the camera_id from the camera document
                    camera_id = camera.get('camera_id', str(camera_doc.id))
                    
                    # Create new stream with visibility flag
                    logger.info(f"Creating stream with type: {camera['input_type']}, value: {camera['input_value']}, id: {camera_id}")
                    stream = CameraStream(
                        camera["input_type"], 
                        camera["input_value"], 
                        visible=is_visible,
                        camera_id=camera_id  # Pass the camera_id
                    )
                    stream.start()
                    active_streams[camera_name] = stream
                    logger.info(f"Stream started for {camera_name}")
                else:
                    # If stream exists but visibility changed, update it
                    current_stream = active_streams[camera_name]
                    if current_stream.is_visible != is_visible:
                        logger.info(f"Updating stream visibility for {camera_name} to {is_visible}")
                        current_stream.set_visibility(is_visible)

            frame_count = 0
            start_time = time.time()
            last_frame_time = time.time()
            
            while True:
                # Check if camera still exists and user still has access
                camera_exists = False
                try:
                    camera_ref = db.collection("cameras").where("camera_name", "==", camera_name).where("owner_uid", "==", user['uid']).limit(1).get()
                    camera_exists = len(list(camera_ref)) > 0
                except Exception as e:
                    logger.error(f"Error checking camera existence: {e}")
                
                if not camera_exists:
                    logger.info(f"Camera {camera_name} no longer exists or user lost access, stopping feed")
                    break

                if camera_name not in active_streams:
                    logger.error(f"Stream {camera_name} not found in active streams")
                    break
                
                frame = active_streams[camera_name].get_frame()
                current_time = time.time()
                
                if frame is None:
                    if current_time - last_frame_time > 5:  # No frames for 5 seconds
                        logger.warning(f"No frames received for {camera_name} in 5 seconds")
                        yield (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n' + 
                              create_error_frame(f"No frames received") + 
                              b'\r\n')
                        break
                    time.sleep(0.1)
                    continue
                
                last_frame_time = current_time
                frame_count += 1
                
                # Log stats occasionally
                if frame_count % 30 == 0:
                    elapsed = current_time - start_time
                    fps = frame_count / elapsed
                    logger.info(f"Stream {camera_name}: delivered {frame_count} frames at {fps:.2f} fps")
                
                try:
                    yield frame
                except Exception as e:
                    logger.error(f"Error yielding frame for {camera_name}: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Video feed error for {camera_name}: {e}")
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + 
                  create_error_frame(f"Stream error: {str(e)}") + 
                  b'\r\n')
        finally:
            # Don't automatically clean up when connection closes
            logger.info(f"Client disconnected from stream {camera_name}")
    
    response = Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Connection'] = 'close'  # Important for some browsers
    return response

@app.route('/camera/<camera_name>/details')
@login_required
def camera_details(camera_name):
    try:
        user = get_current_user()
        if not user:
            return jsonify({"error": "Not authenticated"}), 401
            
        # Verify camera belongs to user
        camera_docs = list(db.collection("cameras")
                        .where("camera_name", "==", camera_name)
                        .where("owner_uid", "==", user['uid'])
                        .limit(1)
                        .stream())
        
        if not camera_docs:
            return jsonify({"error": "Camera not found or access denied"}), 403
            
        camera_data = camera_docs[0].to_dict()
        return jsonify(camera_data)
    except Exception as e:
        logger.error(f"Error getting camera details: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/camera/<camera_name>/seek', methods=['POST'])
@login_required
def seek_video(camera_name):
    try:
        user = get_current_user()
        if not user:
            return jsonify({"success": False, "error": "Not authenticated"}), 401
            
        # Verify camera belongs to user
        camera_docs = list(db.collection("cameras")
                        .where("camera_name", "==", camera_name)
                        .where("owner_uid", "==", user['uid'])
                        .limit(1)
                        .stream())
        
        if not camera_docs:
            return jsonify({"success": False, "error": "Camera not found or access denied"}), 403
            
        data = request.get_json()
        seconds = data.get('seconds', 0)
        
        with stream_lock:
            if camera_name in active_streams:
                # Get current stream info before stopping
                current_stream = active_streams[camera_name]
                was_visible = current_stream.is_visible
                camera_id = current_stream.camera_id  # Save the camera_id
                
                # Stop the current stream
                current_stream.stop()
                
                # Get camera info from database
                camera = camera_docs[0].to_dict()
                
                # Create new stream with the same camera_id
                stream = CameraStream(
                    camera["input_type"], 
                    camera["input_value"], 
                    seek_time=seconds, 
                    visible=was_visible,
                    camera_id=camera_id
                )
                stream.start()
                active_streams[camera_name] = stream
                
                return jsonify({"success": True})
            else:
                return jsonify({"success": False, "error": "Stream not found"}), 404
    except Exception as e:
        logger.error(f"Error seeking video: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/camera/<camera_name>/remove', methods=['POST'])
@login_required
def remove_camera(camera_name):
    if db is None:
        return jsonify({"error": "Firebase not initialized"}), 500

    user = get_current_user()
    if not user:
        return jsonify({"success": False, "error": "Not authenticated"}), 401

    logger.info(f"Starting camera removal process for: {camera_name}")

    try:
        # Verify camera belongs to user
        camera_docs = list(db.collection("cameras")
                        .where("camera_name", "==", camera_name)
                        .where("owner_uid", "==", user['uid'])
                        .limit(1)
                        .stream())
        
        if not camera_docs:
            return jsonify({"success": False, "error": "Camera not found or access denied"}), 403
            
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
        camera_doc = camera_docs[0]
        camera_doc.reference.delete()
        logger.info(f"Deleted camera {camera_name} from database")

        # 3. Force garbage collection multiple times
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

def cleanup_streams():
    """Clean up all active camera streams when the application shuts down"""
    logger.info("Starting global cleanup of all camera streams")
    try:
        with stream_lock:
            camera_names = list(active_streams.keys())
            
            for camera_name in camera_names:
                try:
                    logger.info(f"Force cleaning up stream for camera: {camera_name}")
                    if camera_name in active_streams:
                        stream = active_streams[camera_name]
                        stream.force_stop()
                
                except Exception as e:
                    logger.error(f"Error cleaning up stream for {camera_name}: {e}")
            
            # Clear all streams
            active_streams.clear()
            
            # Force Python garbage collection
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

if __name__ == "__main__":
    if not is_running_from_reloader():
        webbrowser.open("http://127.0.0.1:5000")
    try:
        app.run(debug=True, threaded=True, use_reloader=True)
    finally:
        cleanup_streams()