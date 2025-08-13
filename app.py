import os
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import logging
import traceback
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
class Config:
    BASE_DIR = r"C:\Users\hanik\OneDrive\Documents\college assignments\Brain Stroke Project"
    MODEL_PATH = None  # Will be set after finding the best model
    IMG_SIZE = 224
    OPTIMAL_THRESHOLD = 0.5  # Default threshold, will be updated
    
config = Config()

# Global variables for model and threshold
model = None
optimal_threshold = 0.4  # Lower threshold for medical diagnosis

def find_and_load_model():
    """Find and load the best trained model"""
    global model, optimal_threshold
    
    try:
        # Look for saved models in the base directory
        if not os.path.exists(config.BASE_DIR):
            logger.error(f"Base directory not found: {config.BASE_DIR}")
            return False
            
        model_files = [f for f in os.listdir(config.BASE_DIR) if f.endswith('.h5')]
        
        if not model_files:
            logger.error("No trained models found! Please run the training script first.")
            logger.error(f"Looking in directory: {config.BASE_DIR}")
            logger.error(f"Files found: {os.listdir(config.BASE_DIR)}")
            return False
        
        # Load the first available model
        model_path = os.path.join(config.BASE_DIR, model_files[0])
        logger.info(f"Loading model from: {model_path}")
        
        model = load_model(model_path, compile=False)
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model loaded and compiled successfully!")
        
        # Set medically-focused threshold
        optimal_threshold = 0.4  # Lower threshold to catch more potential strokes
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def preprocess_image(image_data):
    """Preprocess uploaded image for model prediction"""
    try:
        logger.info("Starting image preprocessing...")
        
        # Handle base64 image data
        if isinstance(image_data, str):
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Convert to PIL Image
            image_bytes = base64.b64decode(image_data)
            pil_image = Image.open(BytesIO(image_bytes))
        else:
            # Handle file upload
            pil_image = Image.open(image_data)
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL to numpy array
        img_array = np.array(pil_image)
        
        # Resize image
        img = cv2.resize(img_array, (config.IMG_SIZE, config.IMG_SIZE))
        
        # Enhanced preprocessing for medical images (same as training)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_gray = clahe.apply(img_gray)
        img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        # Normalize
        img = img.astype("float32") / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        logger.info(f"Image preprocessed successfully. Shape: {img.shape}")
        return img
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        logger.error(traceback.format_exc())
        return None

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Handle CORS preflight request
        if request.method == 'OPTIONS':
            return jsonify({'success': True})
        
        logger.info("Received prediction request")
        
        if model is None:
            logger.error("Model not loaded")
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please check server configuration.'
            }), 500
        
        # Get data from request
        if request.content_type == 'application/json':
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({
                    'success': False,
                    'error': 'No image data provided in JSON'
                }), 400
            image_data = data['image']
            
        elif 'file' in request.files:
            # Handle file upload
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400
            image_data = file
            
        else:
            return jsonify({
                'success': False,
                'error': 'No image data provided. Expected JSON with image field or file upload.'
            }), 400
        
        logger.info("Processing image...")
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        
        if processed_image is None:
            return jsonify({
                'success': False,
                'error': 'Failed to process image. Please ensure it is a valid image file.'
            }), 400
        
        # Make prediction
        logger.info("Making prediction...")
        prediction = model.predict(processed_image, verbose=0)[0][0]
        
        # Apply threshold
        has_stroke = prediction > optimal_threshold
        confidence = float(prediction)
        
        logger.info(f"Prediction: {prediction}, Threshold: {optimal_threshold}, Has stroke: {has_stroke}")
        
        # Prepare response
        result = {
            'success': True,
            'has_stroke': bool(has_stroke),
            'confidence': confidence,
            'threshold': optimal_threshold,
            'raw_prediction': float(prediction)
        }
        
        # Add medical interpretation
        if has_stroke:
            result['interpretation'] = f"Potential stroke indicators detected with {confidence:.1%} confidence."
            result['recommendation'] = "Please consult a neurologist or visit the emergency department immediately."
            result['severity'] = "high" if confidence > 0.8 else "medium"
        else:
            result['interpretation'] = f"No significant stroke indicators detected. Model confidence: {(1-confidence):.1%}"
            result['recommendation'] = "Continue regular medical checkups and maintain a healthy lifestyle."
            result['severity'] = "low"
        
        logger.info(f"Returning result: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'optimal_threshold': optimal_threshold,
        'base_dir': config.BASE_DIR,
        'base_dir_exists': os.path.exists(config.BASE_DIR)
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Please upload an image smaller than 16MB.'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        'success': False,
        'error': 'Internal server error. Please try again.'
    }), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("="*60)
    print("BRAIN STROKE DETECTION WEB APPLICATION")
    print("="*60)
    print(f"Base directory: {config.BASE_DIR}")
    print(f"Looking for models in: {config.BASE_DIR}")
    
    # Try to load the model
    if find_and_load_model():
        print("✅ Model loaded successfully!")
        print(f"📊 Optimal threshold: {optimal_threshold}")
        print("🚀 Starting Flask application...")
        print("🌐 Access the application at: http://localhost:5000")
        print("="*60)
        
        # Start the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
        
    else:
        print("❌ Failed to load model!")
        print("\nTroubleshooting steps:")
        print("1. Ensure you have run the training script: python brain_stroke_detection.py")
        print("2. Check that .h5 model files exist in your project directory")
        print(f"3. Verify the base directory path: {config.BASE_DIR}")
        print("4. Make sure the Dataset_labelled folder exists with Normal/ and Stroke/ subfolders")
        print("="*60)