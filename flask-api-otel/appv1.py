# app.py
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import psutil
from datetime import datetime, timedelta
import random
from collections import defaultdict
import threading
import uuid

app = Flask(__name__)

# Configuration
MODELS_DIR = "models"
DEFAULT_MODEL = "best1.pt"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_INFO = {
    "best1.pt": {
        "config": {
            "input_size": [416,416],
            "batch_size": 64,
            "confidence_threshold": 0.4
        },
        "date_registered": "2025-03-13"
    },
    "best2.pt": {
        "config": {
            "input_size": [562, 562],
            "batch_size": 64,
            "confidence_threshold": 0.4
        },
        "date_registered": "2025-03-23"
    }
}

# Global variables
models = {}
current_default_model = DEFAULT_MODEL
start_time = time.time()
metrics = {
    "request_count": 0,
    "request_times": [],
    "model_usage": defaultdict(int),
    "latencies": [],
    "errors": []
}

# Load models
def load_model(model_name):
    if model_name not in models:
        model_path = os.path.join(MODELS_DIR, model_name)
        try:
            models[model_name] = YOLO(model_path)
            print(f"Model {model_name} loaded successfully")
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            return None
    return models[model_name]

# Initialize models
load_model("best1.pt")
load_model("best2.pt")

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(file):
    try:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def format_predictions(results, model_name):
    predictions = []
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            bbox = [int(coord) for coord in xyxy]  # Convert to int
            predictions.append({
                "label": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": bbox
            })
    return {
        "predictions": predictions,
        "model_used": model_name
    }

def get_uptime():
    seconds = int(time.time() - start_time)
    return str(timedelta(seconds=seconds))

def track_metrics(model_name, latency):
    metrics["request_count"] += 1
    metrics["request_times"].append(time.time())
    metrics["model_usage"][model_name] += 1
    metrics["latencies"].append(latency)

def calculate_metrics():
    now = time.time()
    # Calculate request rate per minute
    recent_requests = [t for t in metrics["request_times"] if now - t <= 60]
    request_rate = len(recent_requests)
    
    # Calculate latency metrics
    if metrics["latencies"]:
        avg_latency = sum(metrics["latencies"]) / len(metrics["latencies"])
        max_latency = max(metrics["latencies"])
    else:
        avg_latency = 0
        max_latency = 0
    
    return {
        "request_rate_per_minute": request_rate,
        "avg_latency_ms": round(avg_latency * 1000, 1),
        "max_latency_ms": round(max_latency * 1000, 1),
        "total_requests": metrics["request_count"]
    }

# Alert system
def check_for_alerts():
    if len(metrics["latencies"]) > 10:
        avg_latency = sum(metrics["latencies"][-10:]) / 10
        if avg_latency > 2.5:  # 2.5 second threshold
            return "High latency alert: Average response time over last 10 requests is {:.2f}s".format(avg_latency)
    return None

# Endpoints
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    # Get model parameter (or use default)
    model_name = request.form.get('model', current_default_model)
    
    # A/B testing - 50% chance to use either model if not specified
    if 'model' not in request.form and random.random() < 0.5:
        model_name = "best2.pt" if current_default_model == "best1.pt" else "best1.pt"
    
    # Load model
    model = load_model(model_name)
    if model is None:
        return jsonify({"error": f"Model {model_name} not available"}), 400
    
    try:
        # Process image
        file.seek(0)
        image = process_image(file)
        
        # Run inference
        results = model(image)
        
        # Format response
        response = format_predictions(results, model_name)
        
        # Track metrics
        latency = time.time() - start_time
        track_metrics(model_name, latency)
        
        return jsonify(response)
    
    except Exception as e:
        metrics["errors"].append(str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/health-status', methods=['GET'])
def health_check():
    return jsonify({
        "status": "Healthy",
        "server": "Flask",
        "uptime": get_uptime()
    })

@app.route('/management/models', methods=['GET'])
def list_models():
    return jsonify({
        "available_models": list(MODEL_INFO.keys())
    })

@app.route('/management/models/<model_name>/describe', methods=['GET'])
def describe_model(model_name):
    if model_name not in MODEL_INFO:
        return jsonify({"error": "Model not found"}), 404
    
    return jsonify({
        "model": model_name,
        "config": MODEL_INFO[model_name]["config"],
        "date_registered": MODEL_INFO[model_name]["date_registered"]
    })

@app.route('/management/models/<model_name>/set-default', methods=['GET'])
def set_default_model(model_name):
    global current_default_model
    
    if model_name not in MODEL_INFO:
        return jsonify({"error": "Model not found"}), 404
    
    current_default_model = model_name
    return jsonify({
        "success": True,
        "default_model": current_default_model
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    alert = check_for_alerts()
    metrics_data = calculate_metrics()
    if alert:
        metrics_data["alert"] = alert
    return jsonify(metrics_data)

@app.route('/group-info', methods=['GET'])
def group_info():
    return jsonify({
        "group": "group4",
        "members": ["Sahil", "Chinmay", "Nandhini"]
    })

# Background monitoring thread
def monitor_metrics():
    while True:
        time.sleep(60)
        current_metrics = calculate_metrics()
        print(f"\n[Monitoring] Current metrics: {current_metrics}")
        
        alert = check_for_alerts()
        if alert:
            print(f"[ALERT] {alert}")

# Start monitoring thread
monitor_thread = threading.Thread(target=monitor_metrics, daemon=True)
monitor_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)