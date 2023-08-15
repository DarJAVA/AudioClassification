from flask import Flask, render_template, request, jsonify
import os
import librosa
import numpy as np
import tensorflow as tf
import model

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('test_model.h5')  

# Function to extract features from audio using librosa
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    return mfccs

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    if audio_file.filename == '':
        return jsonify({'error': 'No selected audio file'}), 400

    if audio_file:
        audio_path = os.path.join('uploads', audio_file.filename)
        audio_file.save(audio_path)
        features = extract_features(audio_path)
        features = np.expand_dims(features, axis=0)

        # Call the model for prediction
        prediction = model.predict(features)
        class_label = np.argmax(prediction)

        # Define class labels 
        class_labels = ['class_country', 'class_hiphop', 'class_metal', 'class_pop', 'class_rock']

        return jsonify({'predicted_class': class_labels[class_label]})

if __name__ == '__main__':
    app.run(debug=True)

