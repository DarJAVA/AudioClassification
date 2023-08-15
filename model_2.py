import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers

# Function to extract features from audio using librosa
def extract_features(file_path):
    y, _ = librosa.load(file_path, sr=None)
    return y

#max_length=num num=number of how long each of data
def prepare_dataset(data_folder, max_length=10000):
    X = []
    y = []
    for class_label, class_name in enumerate(os.listdir(data_folder)):
        class_folder = os.path.join(data_folder, class_name)
        for file_name in os.listdir(class_folder):
            if file_name.endswith('.wav'):
                file_path = os.path.join(class_folder, file_name)
                waveform = extract_features(file_path)

                # Apply padding or truncation to the fixed max_length
                if max_length is not None:
                    if len(waveform) < max_length:
                        waveform = np.pad(waveform, (0, max_length - len(waveform)))
                    else:
                        waveform = waveform[:max_length]

                X.append(waveform)
                y.append(class_label)

    X = np.array(X)
    y = np.array(y)

    return X, y

# Load your own dataset and split it into training and testing sets
data_folder = 'my_audio_dataset'
X, y = prepare_dataset(data_folder)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Build the model
model = tf.keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
# Replace num_classes with the number of classes in your dataset (layers.Dense(num_classes,'sigmoid')) 
    layers.Dense(5, activation='sigmoid')  
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')