from flask import Flask, request, jsonify
import cv2
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load your pre-trained model (update the path)
model = tf.keras.models.load_model('akshay.h5')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to the model's input size
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize
    return image

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Save the uploaded file and preprocess
    file_path = "uploads/" + file.filename
    file.save(file_path)
    image = preprocess_image(file_path)
    
    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)
    
    # Return the result
    return jsonify({'landmark': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
