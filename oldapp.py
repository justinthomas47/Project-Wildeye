import os
import webbrowser
from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)

# Firebase Admin SDK Initialization
cred = credentials.Certificate("C:\\Users\\JUSTIN THOMAS\\Desktop\\Project web app\\firebase_admin_key.json")  # Replace with your Firebase Admin SDK JSON file
firebase_admin.initialize_app(cred)
db = firestore.client()

# Route: Index (Login/Signup Page)
@app.route("/")
def index():
    return render_template("index.html")

# Route: Home (Camera Input Page)
@app.route("/home", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        camera_name = request.form["camera_name"]
        rtsp_url = request.form["rtsp_url"]
        address = request.form["address"]
        latitude = request.form.get("latitude", "")
        longitude = request.form.get("longitude", "")

        # Save camera details to Firebase Firestore
        camera_data = {
            "camera_name": camera_name,
            "rtsp_url": rtsp_url,
            "address": address,
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        db.collection("cameras").add(camera_data)

        return redirect(url_for("home"))

    # Fetch all cameras from Firestore to display on the page
    cameras_ref = db.collection("cameras")
    cameras = [doc.to_dict() for doc in cameras_ref.stream()]
    return render_template("home.html", cameras=cameras)

# Route: About Page
@app.route("/about")
def about():
    return render_template("about.html")

# Route: Contact Page
@app.route("/contact")
def contact():
    return render_template("contact.html")

# Route: FAQ Page
@app.route("/faq")
def faq():
    return render_template("faq.html")

# API: Save Detection Logs
@app.route("/save_detection", methods=["POST"])
def save_detection():
    data = request.get_json()
    log_data = {
        "animal": data.get("animal"),
        "camera_name": data.get("camera_name"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save detection log to Firestore
    db.collection("detection_logs").add(log_data)
    return jsonify({"success": True, "message": "Detection log saved."})

# Start the Flask app
if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True) 