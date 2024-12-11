from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import json
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)


# Load the saved model
model_path = 'models/my_model.keras'
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# Load the class indices (mapping from class labels to indices)
class_indices_path = 'models/class_indices.json'
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)
label_map = {v: k for k, v in class_indices.items()}

# Preprocess the image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Ensure the uploads directory exists
os.makedirs("uploads", exist_ok=True)

# File path for the Excel sheet
excel_file_path = "product_predictions.xlsx"

# Create or load the Excel file
def initialize_excel():
    if not os.path.exists(excel_file_path):
        df = pd.DataFrame(columns=['S. No.', 'File Name', 'Predicted Brand', 'Timestamp'])
        df.to_excel(excel_file_path, index=False)

# Function to update Excel with new data
def update_excel(data):
    if os.path.exists(excel_file_path):
        df = pd.read_excel(excel_file_path)
    else:
        df = pd.DataFrame(columns=['S. No.', 'File Name', 'Predicted Brand', 'Timestamp'])

    new_row = pd.DataFrame(data, columns=df.columns)
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(excel_file_path, index=False)

# Home route
@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have an index.html in your templates folder


# Prediction API
@app.route('/predict-folder', methods=['POST'])
def predict_folder():
    if 'images' not in request.files:
        return jsonify({"error": "No files found in the request"}), 400

    images = request.files.getlist('images')
    results = []
    excel_data = []

    # Loop through each file
    for idx, img_file in enumerate(images, start=1):
        filename = secure_filename(img_file.filename)
        file_path = os.path.join('uploads', filename)
        img_file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Predict using the model
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = label_map[predicted_class]

        # Get current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prediction result
        result = {
            'predicted_brand': predicted_label
        }
        results.append(result)

        # Prepare data for Excel
        excel_data.append([
            len(results),  # S. No.
            filename,  # File Name
            predicted_label,  # Predicted Brand
            current_time  # Timestamp
        ])

        # Clean up the uploaded file (optional)
        os.remove(file_path)

    # Update Excel
    update_excel(excel_data)

    return jsonify(results)

# Endpoint to download the Excel file
@app.route('/download-excel', methods=['GET'])
def download_excel():
    if os.path.exists(excel_file_path):
        return send_file(excel_file_path, as_attachment=True)
    else:
        return jsonify({"error": "Excel file not found"}), 404
        
if __name__ == '__main__':
    app.run(debug=True)
