import os
import cv2
import json
import time
import numpy as np
from dotenv import load_dotenv
from roboflow import Roboflow
import matplotlib.pyplot as plt
from utils import plot_confusion_matrix

load_dotenv()

def initialize_model():
    """
    Initializes the Roboflow model using an API key from environment variables.

    Returns:
        model: The loaded Roboflow model.
    """
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("API key for Roboflow is not set. Please set ROBOFLOW_API_KEY as an environment variable.")

    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project("newcassava")
        model = project.version("3").model
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def draw_bounding_boxes(img, predictions):
    """
    Draws bounding boxes and labels on the image for each prediction.

    Args:
        img (ndarray): The image to draw on.
        predictions (list): List of prediction dictionaries.

    Returns:
        ndarray: The image with bounding boxes and labels.
    """
    total_confidence = 0
    num_predictions = len(predictions)

    for prediction in predictions:
        disease_name = prediction.get("class", "Unknown")
        confidence = prediction.get("confidence", 0)
        x0 = int(prediction["x"] - prediction["width"] / 2)
        y0 = int(prediction["y"] - prediction["height"] / 2)
        x1 = int(prediction["x"] + prediction["width"] / 2)
        y1 = int(prediction["y"] + prediction["height"] / 2)

        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 8)

        label = f"{disease_name} ({confidence:.2f})"
        cv2.putText(img, label, (x0, y0 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4, cv2.LINE_AA)

        total_confidence += confidence

    avg_confidence = (total_confidence / num_predictions) if num_predictions else 0
    accuracy_label = f"Accuracy: {avg_confidence * 100:.2f}%"
    cv2.putText(img, accuracy_label, (10, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4, cv2.LINE_AA)

    return img

def resize_image(img, max_width=800):
    """
    Resizes the image to a maximum width while maintaining the aspect ratio.

    Args:
        img (ndarray): The original image.
        max_width (int): The maximum width for the resized image.

    Returns:
        ndarray: The resized image.
    """
    height, width = img.shape[:2]
    if width > max_width:
        scaling_factor = max_width / float(width)
        new_dim = (max_width, int(height * scaling_factor))
        return cv2.resize(img, new_dim)
    return img

def plot_model_history(model_history):
    """Plot Accuracy and Loss curves given the model history."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'val'], loc='best')

    # Loss plot
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'val'], loc='best')

    fig.savefig('public/plot-cassava.png')
    plt.show()

def plot_model_history(model_history):
    """
    Plot accuracy and loss curves using model training history.

    Args:
        model_history: History object from model training (e.g., TensorFlow/Keras).
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='best')

    # Plot loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='best')

    plt.tight_layout()
    fig.savefig('public/plot-cassava.png')
    plt.show()

if __name__ == "__main__":
    model = initialize_model()
    if model:
        # Example training process
        # Assuming your model is compatible with TensorFlow/Keras fit function
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense

        # Mock dataset
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(2, size=(100, 1))
        X_val = np.random.rand(20, 10)
        y_val = np.random.randint(2, size=(20, 1))

        # Example model
        keras_model = Sequential([
            Dense(64, activation='relu', input_shape=(10,)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history = keras_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=8)

        # Plot the history
        plot_model_history(history)
