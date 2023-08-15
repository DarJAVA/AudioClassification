import os
import librosa
import numpy as np
import tensorflow as tf

# Function to extract features from audio using librosa
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    return mfccs

# Prepare the dataset for testing
def prepare_dataset(data_folder):
    X = []
    y = []
    for class_label, class_name in enumerate(os.listdir(data_folder)):
        class_folder = os.path.join(data_folder, class_name)
        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)
            try:
                if os.path.isfile(file_path) and file_name.endswith('.wav'):
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(class_label)
            except Exception as e:
                print(f"Error processing file: {file_path}")
                print(e)

    X = np.array(X)
    y = np.array(y)

    return X, y

# Load your own dataset and extract features for testing 
test_folder = 'my_audio_dataset'
X_test_new, y_test_new = prepare_dataset(test_folder)

# Load the trained model 
model = tf.keras.models.load_model('your_model.h5')

# Predict on the test data
predictions = model.predict(X_test_new)
predicted_labels = np.argmax(predictions, axis=1)  # Get the class index with the highest probability

# Calculate accuracy
accuracy = np.mean(predicted_labels == y_test_new)
print(f'Test accuracy on new data: {accuracy:.4f}')

model.save('test_model.h5')