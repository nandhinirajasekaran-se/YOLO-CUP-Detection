from flask import Flask, request, jsonify, send_file
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
import json
import matplotlib.pyplot as plt
import pandas as pd
import io
from collections import defaultdict
from flask_cors import CORS

# OpenTelemetry tracing
from opentelemetry import trace, metrics
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# OpenTelemetry metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader


app = Flask(__name__)
CORS(app)

# --- Tracing Setup ---
resource = Resource(attributes={
    SERVICE_NAME: "flask-api-service"
})

trace.set_tracer_provider(TracerProvider(resource=resource))
tracer_provider = trace.get_tracer_provider()

span_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4318/v1/traces")
span_processor = BatchSpanProcessor(span_exporter)
tracer_provider.add_span_processor(span_processor)

# Instrument Flask for tracing
FlaskInstrumentor().instrument_app(app)

# --- Metrics Setup ---
metric_exporter = OTLPMetricExporter(endpoint="http://otel-collector:4318/v1/metrics")

metric_reader = PeriodicExportingMetricReader(metric_exporter)
metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

meter = metrics.get_meter("flask-api-meter")
request_counter = meter.create_counter(
    "http_requests_total",
    description="Total HTTP requests"
)



# Configuration
MODELS_DIR = "models"
METRICS_DIR = "metrics_data"
os.makedirs(METRICS_DIR, exist_ok=True)
DEFAULT_MODEL = "best1.pt"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ANNOTATIONS_DIR = "annotations"
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

MODEL_INFO = {
    "best2.pt": {
        "config": {
            "input_size": [416,416],
            "batch_size": 64,
            "confidence_threshold": 0.4
        },
        "date_registered": "2025-03-13",
        "metrics": {
            "precision": [],
            "recall": [],
            "map50": [],
            "map50_95": [],
            "accuracy": [],
            "class_metrics": defaultdict(dict)
        }
    },
    "best1.pt": {
        "config": {
            "input_size": [562, 562],
            "batch_size": 64,
            "confidence_threshold": 0.4
        },
        "date_registered": "2025-03-23",
        "metrics": {
            "precision": [],
            "recall": [],
            "map50": [],
            "map50_95": [],
            "accuracy": [],
            "class_metrics": defaultdict(dict)
        }
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

# Helper functions
def load_model(model_name):
    if model_name not in models:
        model_path = os.path.join(MODELS_DIR, model_name)
        try:
            models[model_name] = YOLO(model_path, task='detect')
            print(f"Model {model_name} loaded successfully")
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            return None
    return models[model_name]

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
            bbox = [int(coord) for coord in xyxy]
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

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def read_yolo_annotations(annotation_path, img_width, img_height):
    boxes = []
    if not os.path.exists(annotation_path):
        return boxes
        
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            class_id, x_center, y_center, width, height = map(float, parts[:5])
            
            x1 = (x_center - width/2) * img_width
            y1 = (y_center - height/2) * img_height
            x2 = (x_center + width/2) * img_width
            y2 = (y_center + height/2) * img_height
            
            boxes.append({
                'class_id': int(class_id),
                'bbox': [x1, y1, x2, y2],
                'matched': False
            })
    return boxes

def evaluate_predictions(ground_truth, predictions, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for pred in predictions:
        best_iou = 0
        best_match = None
        
        for gt in ground_truth:
            if gt['matched'] or gt['class_id'] != pred['class_id']:
                continue
                
            iou = calculate_iou(gt['bbox'], pred['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_match = gt
                
        if best_iou >= iou_threshold and best_match:
            true_positives += 1
            best_match['matched'] = True
            class_stats[pred['class_id']]['tp'] += 1
        else:
            false_positives += 1
            class_stats[pred['class_id']]['fp'] += 1
    
    false_negatives = sum(1 for gt in ground_truth if not gt['matched'])
    for gt in ground_truth:
        if not gt['matched']:
            class_stats[gt['class_id']]['fn'] += 1
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'class_stats': dict(class_stats)
    }

def calculate_map(model, dataset_path):
    """Calculate mAP metrics using YOLO's native validation"""
    try:
        # Create temporary dataset.yaml
        with open('temp_dataset.yaml', 'w') as f:
            f.write(f"""
            path: {os.path.abspath(dataset_path)}
            train: images
            val: images
            names: {model.names}
            """)
        
        # Run validation
        results = model.val(data='temp_dataset.yaml', split='val')
        
        return {
            'map50': results.results_dict['metrics/mAP50(B)'],
            'map50_95': results.results_dict['metrics/mAP50-95(B)']
        }
    except Exception as e:
        print(f"Error calculating mAP: {str(e)}")
        return {'map50': None, 'map50_95': None}

def log_model_metrics(model_name, metrics_data):
    timestamp = datetime.now().isoformat()
    
    for metric in ["precision", "recall", "map50", "map50_95", "accuracy"]:
        if metric in metrics_data and metrics_data[metric] is not None:
            MODEL_INFO[model_name]["metrics"][metric].append({
                "value": metrics_data[metric],
                "timestamp": timestamp
            })
    
    if "class_metrics" in metrics_data:
        for class_name, class_data in metrics_data["class_metrics"].items():
            for metric, value in class_data.items():
                if metric not in MODEL_INFO[model_name]["metrics"]["class_metrics"][class_name]:
                    MODEL_INFO[model_name]["metrics"]["class_metrics"][class_name][metric] = []
                MODEL_INFO[model_name]["metrics"]["class_metrics"][class_name][metric].append({
                    "value": value,
                    "timestamp": timestamp
                })
    
    metrics_file = os.path.join(METRICS_DIR, f"{model_name}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(MODEL_INFO[model_name]["metrics"], f)

def get_model_metrics(model_name, time_range_hours=24):
    if model_name not in MODEL_INFO:
        return None
    
    cutoff = datetime.now() - timedelta(hours=time_range_hours)
    
    metrics = {}
    for metric in ["precision", "recall", "map50", "map50_95", "accuracy"]:
        metrics[metric] = [
            m for m in MODEL_INFO[model_name]["metrics"][metric] 
            if datetime.fromisoformat(m["timestamp"]) >= cutoff
        ]
    
    metrics["class_metrics"] = {}
    for class_name, class_data in MODEL_INFO[model_name]["metrics"]["class_metrics"].items():
        metrics["class_metrics"][class_name] = {}
        for metric, values in class_data.items():
            metrics["class_metrics"][class_name][metric] = [
                v for v in values 
                if datetime.fromisoformat(v["timestamp"]) >= cutoff
            ]
    
    return metrics

def generate_metrics_plot(model_name, metric_type, time_range_hours=24):
    metrics = get_model_metrics(model_name, time_range_hours)
    if not metrics or not metrics.get(metric_type):
        return None
    
    timestamps = [m["timestamp"] for m in metrics[metric_type]]
    values = [m["value"] for m in metrics[metric_type]]
    
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, values, marker='o')
    plt.title(f"{metric_type.upper()} for {model_name}")
    plt.xlabel("Time")
    plt.ylabel(metric_type.upper())
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return buf

def calculate_metrics():
    now = time.time()
    
    recent_requests = [t for t in metrics["request_times"] if now - t <= 60]
    request_rate = len(recent_requests)
    
    if metrics["latencies"]:
        avg_latency = sum(metrics["latencies"]) / len(metrics["latencies"])
        max_latency = max(metrics["latencies"])
        min_latency = min(metrics["latencies"])
    else:
        avg_latency = max_latency = min_latency = 0
    
    total_requests = metrics["request_count"]
    error_rate = len(metrics["errors"]) / total_requests if total_requests > 0 else 0
    
    return {
        "request_rate_per_minute": request_rate,
        "avg_latency_ms": round(avg_latency * 1000, 1),
        "max_latency_ms": round(max_latency * 1000, 1),
        "min_latency_ms": round(min_latency * 1000, 1),
        "total_requests": total_requests,
        "error_rate": round(error_rate, 3),
        "model_usage": dict(metrics["model_usage"])
    }

def check_for_alerts():
    if len(metrics["latencies"]) > 10:
        avg_latency = sum(metrics["latencies"][-10:]) / 10
        if avg_latency > 2.5:
            return {
                "type": "high_latency",
                "message": f"High latency detected (avg: {avg_latency:.2f}s over last 10 requests)",
                "severity": "warning"
            }
    
    if metrics["request_count"] > 20 and len(metrics["errors"]) / metrics["request_count"] > 0.1:
        return {
            "type": "high_error_rate",
            "message": "Error rate exceeds 10%",
            "severity": "critical"
        }
    
    return None

# Initialize models
for model_name in MODEL_INFO:
    load_model(model_name)

# Endpoints
@app.route('/predict', methods=['POST'])
def predict():
    request_counter.add(1, attributes={"endpoint": "/predict", "method": "POST"})
    start_time = time.time()
    
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    model_name = request.form.get('model', current_default_model)
    model = load_model(model_name)
    if model is None:
        return jsonify({"error": f"Model {model_name} not available"}), 400
    
    try:
        file.seek(0)
        image = process_image(file)
        img_height, img_width = image.shape[:2]
        
        results = model(image)
        
        predictions = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                predictions.append({
                    'class_id': int(box.cls[0]),
                    'bbox': [int(coord) for coord in xyxy],
                    'confidence': float(box.conf[0])
                })
        
        annotation_path = os.path.join(ANNOTATIONS_DIR, os.path.splitext(file.filename)[0] + '.txt')
        metrics_data = None
        
        ground_truth = read_yolo_annotations(annotation_path, img_width, img_height)
        if ground_truth:
            eval_results = evaluate_predictions(ground_truth, predictions)
            
            metrics_data = {
                'precision': eval_results['precision'],
                'recall': eval_results['recall'],
                'class_metrics': {
                    str(class_id): {
                        'precision': stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0,
                        'recall': stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
                    }
                    for class_id, stats in eval_results['class_stats'].items()
                }
            }
            
            log_model_metrics(model_name, metrics_data)
        
        response = format_predictions(results, model_name)
        latency = time.time() - start_time
        track_metrics(model_name, latency)
        
        return jsonify(response)
    
    except Exception as e:
        metrics["errors"].append(str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    request_counter.add(1, attributes={"endpoint": "/evaluate", "method": "POST"})

    if 'model' not in request.form:
        return jsonify({"error": "Model not specified"}), 400
    
    model_name = request.form['model']
    model = load_model(model_name)
    if model is None:
        return jsonify({"error": f"Model {model_name} not available"}), 400
    
    dataset_path = request.form.get('dataset_path', 'data/eval')
    images_dir = os.path.join(dataset_path, 'images')
    annotations_dir = os.path.join(dataset_path, 'labels')
    
    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        return jsonify({"error": "Invalid dataset path"}), 400
    
    try:
        # Calculate precision/recall per image
        total_tp = 0
        total_fp = 0
        total_fn = 0
        class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        processed_files = 0
        for img_file in os.listdir(images_dir):
            if not allowed_file(img_file):
                continue
                
            img_path = os.path.join(images_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            img_height, img_width = image.shape[:2]
            
            results = model(image)
            predictions = []
            for result in results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].tolist()
                    predictions.append({
                        'class_id': int(box.cls[0]),
                        'bbox': [int(coord) for coord in xyxy],
                        'confidence': float(box.conf[0])
                    })
            
            annotation_file = os.path.splitext(img_file)[0] + '.txt'
            annotation_path = os.path.join(annotations_dir, annotation_file)
            
            ground_truth = read_yolo_annotations(annotation_path, img_width, img_height)
            if ground_truth:
                eval_results = evaluate_predictions(ground_truth, predictions)
                
                total_tp += eval_results['true_positives']
                total_fp += eval_results['false_positives']
                total_fn += eval_results['false_negatives']
                
                for class_id, stats in eval_results['class_stats'].items():
                    class_stats[class_id]['tp'] += stats['tp']
                    class_stats[class_id]['fp'] += stats['fp']
                    class_stats[class_id]['fn'] += stats['fn']
                
                processed_files += 1
        
        if processed_files == 0:
            return jsonify({"error": "No valid images processed"}), 400
        
        # Calculate mAP metrics
        map_metrics = calculate_map(model, dataset_path)
        
        # Calculate final metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        response = {
            'model': model_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'map50': map_metrics['map50'],
            'map50_95': map_metrics['map50_95'],
            'processed_files': processed_files,
            'class_metrics': {
                str(class_id): {
                    'precision': stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0,
                    'recall': stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
                }
                for class_id, stats in class_stats.items()
            }
        }
        
        log_model_metrics(model_name, {
            'precision': precision,
            'recall': recall,
            'map50': map_metrics['map50'],
            'map50_95': map_metrics['map50_95'],
            'class_metrics': {
                str(class_id): {
                    'precision': stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0,
                    'recall': stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
                }
                for class_id, stats in class_stats.items()
            }
        })
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_system_metrics():
    request_counter.add(1, attributes={"endpoint": "/metrics", "method": "GET"})

    alert = check_for_alerts()
    metrics_data = calculate_metrics()
    
    response = {
        "system_metrics": metrics_data,
        "timestamp": datetime.now().isoformat()
    }
    
    if alert:
        response["alert"] = alert
    
    return jsonify(response)

@app.route('/metrics/models/<model_name>', methods=['GET'])
def get_model_metrics_endpoint(model_name):
    request_counter.add(1, attributes={"endpoint": "/metrics/models/"+model_name, "method": "GET"})

    if model_name not in MODEL_INFO:
        return jsonify({"error": "Model not found"}), 404
    
    time_range = request.args.get('time_range', default=24, type=int)
    metrics = get_model_metrics(model_name, time_range)
    
    return jsonify({
        "model": model_name,
        "metrics": metrics
    })

@app.route('/metrics/models/<model_name>/plot', methods=['GET'])
def get_model_metrics_plot(model_name):
    request_counter.add(1, attributes={"endpoint": "/metrics/models/"+model_name, "method": "GET"})
    if model_name not in MODEL_INFO:
        return jsonify({"error": "Model not found"}), 404
    
    metric_type = request.args.get('metric', default="precision")
    time_range = request.args.get('time_range', default=24, type=int)
    
    buf = generate_metrics_plot(model_name, metric_type, time_range)
    if not buf:
        return jsonify({"error": "No data available"}), 404
    
    return send_file(buf, mimetype='image/png')

@app.route('/metrics/compare', methods=['GET'])
def compare_models():
    request_counter.add(1, attributes={"endpoint": "/metrics/compare", "method": "GET"})

    comparison = {}
    time_range = request.args.get('time_range', default=24, type=int)
    
    for model_name in MODEL_INFO:
        metrics = get_model_metrics(model_name, time_range)
        if metrics:
            summary = {}
            for metric in ["precision", "recall", "map50", "map50_95"]:
                if metrics[metric]:
                    values = [m["value"] for m in metrics[metric]]
                    summary[f"avg_{metric}"] = sum(values) / len(values) if values else None
                    summary[f"latest_{metric}"] = metrics[metric][-1]["value"] if metrics[metric] else None
            
            comparison[model_name] = summary
    
    return jsonify(comparison)

@app.route('/management/models', methods=['GET'])
def list_models():
    request_counter.add(1, attributes={"endpoint": "/management/models", "method": "GET"})

    return jsonify({
        "available_models": list(MODEL_INFO.keys())
    })

@app.route('/management/models/<model_name>/describe', methods=['GET'])
def describe_model(model_name):
    request_counter.add(1, attributes={"endpoint": "/management/models/"+model_name+"/describe", "method": "GET"})
    if model_name not in MODEL_INFO:
        return jsonify({"error": "Model not found"}), 404
    
    time_range = request.args.get('time_range', default=24, type=int)
    metrics = get_model_metrics(model_name, time_range)
    
    metrics_summary = {}
    for metric in ["precision", "recall", "map50", "map50_95", "accuracy"]:
        if metrics[metric]:
            metrics_summary[metric] = metrics[metric][-1]["value"] if metrics[metric] else None
        else:
            metrics_summary[metric] = None
    
    return jsonify({
        "model": model_name,
        "config": MODEL_INFO[model_name]["config"],
        "date_registered": MODEL_INFO[model_name]["date_registered"],
        "metrics_summary": metrics_summary
    })

@app.route('/management/models/<model_name>/set-default', methods=['GET'])
def set_default_model(model_name):
    request_counter.add(1, attributes={"endpoint": "/management/models/"+model_name+"/set-default", "method": "GET"})
    global current_default_model
    
    if model_name not in MODEL_INFO:
        return jsonify({"error": "Model not found"}), 404
    
    current_default_model = model_name
    return jsonify({
        "success": True,
        "default_model": current_default_model
    })

@app.route('/health-status', methods=['GET'])
def health_check():
    request_counter.add(1, attributes={"endpoint": "/health-status", "method": "GET"})
    return jsonify({
        "status": "Healthy",
        "server": "Flask",
        "uptime": get_uptime()
    })

@app.route('/group-info', methods=['GET'])
def group_info():
    request_counter.add(1, attributes={"endpoint": "/group-info", "method": "GET"})
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
    app.run(host='0.0.0.0', port=6034, threaded=True)