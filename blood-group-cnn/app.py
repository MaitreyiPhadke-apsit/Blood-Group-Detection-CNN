from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import tensorflow as tf
import cv2
import os
import ssl
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Disable SSL verification if needed for downloading models
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize Flask app
app = Flask(__name__)

# Define Upload Folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model
MODEL_PATH = os.path.join('model', 'model.h5')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found! Ensure 'model.h5' exists in the 'model' folder.")
model = tf.keras.models.load_model(MODEL_PATH)

# Define the allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess Image for Model
def preprocess_image(file_path):
    try:
        img = load_img(file_path, target_size=(64, 64))  # Ensure the input size matches the model
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        raise RuntimeError(f"Error preprocessing image: {str(e)}")

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

# Serve Uploaded Images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Predict Blood Group from Fingerprint Image
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types are png, jpg, jpeg, bmp'}), 400

    # Save the uploaded file securely
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        # Preprocess the image
        img = preprocess_image(file_path)

        # Perform prediction
        predictions = model.predict(img)
        predicted_class = int(np.argmax(predictions[0]))

        # Define class names based on model training
        class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
        if predicted_class >= len(class_names):
            return jsonify({'error': 'Prediction index out of bounds'}), 500

        predicted_label = class_names[predicted_class]

        # Return JSON response with correct image path
        return jsonify({
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': float(np.max(predictions[0])),
            'uploaded_image': f"/uploads/{filename}"  # Correct path for frontend
        })

    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
