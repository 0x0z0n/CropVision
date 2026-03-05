import os
import sys
import signal
import subprocess
import uuid
import io
from pathlib import Path
import numpy as np
import tensorflow as tf
import webbrowser
from threading import Timer

# Security & Image Processing
from PIL import Image, UnidentifiedImageError

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template

# --- PORT CLEANUP LOGIC ---
def revoke_previous_sessions(port):
    """Finds and kills any process currently using the specified port."""
    try:
        pid = subprocess.check_output(["lsof", "-t", f"-i:{port}"]).decode().strip()
        if pid:
            print(f"[!] Port {port} is occupied by PID {pid}. Revoking session...")
            os.kill(int(pid), signal.SIGKILL)
    except subprocess.CalledProcessError:
        pass
    except Exception as e:
        print(f"Cleanup Error: {e}")

# 1. Modern TF 2.x GPU Memory Configuration
if "LD_LIBRARY_PATH" not in os.environ:
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and memory growth is enabled.")
    except RuntimeError as e:
        print(f"GPU Config Error: {e}")
else:
    print("GPU not detected. TensorFlow will run on CPU.")

app = Flask(__name__)

# --- SECURITY CONFIGURATIONS ---
# Limit upload size to 5 Megabytes to prevent DoS attacks
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_extension(filename):
    """Performs a surface-level check of the file extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_image(file_stream):
    """
    Reads the file stream to verify the actual file signature (magic numbers)
    and structural integrity, completely ignoring the client-provided MIME type.
    """
    try:
        # Read the first chunk of the file into memory
        header = file_stream.read(512)
        file_stream.seek(0) # Reset the file pointer back to the beginning!
        
        # Attempt to open the header as an image using Pillow
        img = Image.open(io.BytesIO(header))
        
        # Verify the detected format is in our allowed list
        if img.format.lower() not in ALLOWED_EXTENSIONS:
            return False
        return True
    except (UnidentifiedImageError, Exception):
        # If Pillow can't recognize it as an image, it's malicious or corrupted.
        return False

# --- PATH & MODEL SETUP ---
BASE_DIR = Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / 'tomato_disease48.h5'

if not MODEL_PATH.exists():
    print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")
    sys.exit(1)

print(f"Loading model from {MODEL_PATH}...")
model = load_model(str(MODEL_PATH), compile=False)
print("Model loaded successfully.")

CLASSES = [
    "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
    "Septoria_leaf_spot", "Spider_mites Two-spotted_spider_mite",
    "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "mosaic_virus", "Healthy"
]

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(48, 48))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    pred_class_index = np.argmax(preds, axis=1)[0]
    return CLASSES[pred_class_index] if pred_class_index < len(CLASSES) else "Unknown/Healthy"

# --- ROUTES ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part in the request", 400
    
    f = request.files['file']
    
    if f.filename == '':
        return "No file selected for uploading", 400

    # SECURITY CHECK 1: Validate extension
    if not allowed_extension(f.filename):
        return "Invalid file extension. Only JPG and PNG allowed.", 400

    # SECURITY CHECK 2: Deep content validation (Signature check)
    if not is_valid_image(f):
        return "File contents are invalid or corrupted. Not a recognized image.", 400

    uploads_dir = BASE_DIR / 'uploads'
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    # SECURITY CHECK 3: Discard user filename, generate a secure UUID
    ext = f.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    file_path = uploads_dir / unique_filename
    
    try:
        # Save and predict
        f.save(str(file_path))
        result = model_predict(str(file_path), model)
        return result
    except Exception as e:
        print(f"!!! SERVER ERROR: {str(e)}")
        return "An error occurred during image processing.", 500
    finally:
        # SECURITY CHECK 4: Guaranteed cleanup even if model_predict fails
        if file_path.exists():
            file_path.unlink()

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5001/')

if __name__ == '__main__':
    revoke_previous_sessions(5001)
    Timer(1.5, open_browser).start()
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
