import os
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
import io

# ... [keep your existing setup and imports] ...

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def is_valid_image(file_stream):
    """
    Reads the file stream to verify the actual file signature (magic numbers)
    and structural integrity, ignoring the client-provided MIME type.
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
            
        # Optional: You can also use img.verify() for a deeper check, 
        # but checking the format of the header is usually sufficient to block scripts.
        return True
        
    except (UnidentifiedImageError, Exception):
        # If Pillow can't recognize it as an image, it's malicious or corrupted.
        return False

def allowed_extension(filename):
    """Only checks the extension for basic filtering."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part in the request", 400
    
    f = request.files['file']
    
    if f.filename == '':
        return "No file selected", 400

    # 1. Quick surface-level check (Extension)
    if not allowed_extension(f.filename):
        return "Invalid file extension.", 400

    # 2. Deep content-level check (The Real Defense)
    if not is_valid_image(f):
        return "File contents are invalid or corrupted. Not a real image.", 400

    uploads_dir = BASE_DIR / 'uploads'
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Secure the filename (prevents directory traversal like "../../etc/passwd")
    safe_filename = secure_filename(f.filename)
    file_path = uploads_dir / safe_filename
    
    try:
        f.save(str(file_path))
        result = model_predict(str(file_path), model)
        return result
    except Exception as e:
        print(f"!!! SERVER ERROR: {str(e)}")
        return "An error occurred during image processing.", 500
    finally:
        if file_path.exists():
            file_path.unlink()
