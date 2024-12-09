import os
import cv2
import json
import time
import numpy as np
from dotenv import load_dotenv
from roboflow import Roboflow
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

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

def evaluate_model(keras_model, X_test, y_test):
    """
    Evaluates the model performance on the test dataset.

    Args:
        keras_model: The trained Keras model.
        X_test: The test input data.
        y_test: The test labels.

    Returns:
        None
    """
    test_loss, test_accuracy = keras_model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

def save_model(keras_model, model_name="cassava_model.keras"):
    """
    Saves the trained model to disk.

    Args:
        keras_model: The trained Keras model.
        model_name: The name of the file where the model will be saved.

    Returns:
        None
    """
    keras_model.save(model_name)
    print(f"Model saved as {model_name}")

def load_trained_model(model_path="cassava_model.keras"):
    """
    Loads a previously trained model from disk.

    Args:
        model_path: Path to the saved model file.

    Returns:
        keras.Model: The loaded model.
    """
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        print(f"Model file '{model_path}' not found!")
        return None

def visualize_predictions(model, X_test, y_test, num_samples=5):
    """
    Visualizes the model predictions on sample test data.

    Args:
        model: The trained model.
        X_test: The test input data.
        y_test: The test labels.
        num_samples: Number of random samples to display.

    Returns:
        None
    """
    random_indices = np.random.choice(range(X_test.shape[0]), num_samples)
    
    for i in random_indices:
        sample = X_test[i]
        true_label = y_test[i]
        
        # Reshape sample if necessary, assuming image input
        # Since your data seems to be 1D, we will visualize it differently
        sample_reshaped = sample.reshape(-1, 1) 

        # For simplicity, visualize this as a bar chart or a simple line plot
        plt.figure(figsize=(5, 3))
        plt.bar(range(len(sample)), sample)  # If you prefer a line plot, use plt.plot(sample)
        plt.title(f"Sample {i}: True Label = {true_label}")
        plt.xlabel("Feature Index")
        plt.ylabel("Feature Value")
        plt.show()

        # Predict the label
        prediction = model.predict(sample.reshape(1, -1))
        predicted_label = 1 if prediction >= 0.5 else 0  # For binary classification

        print(f"Sample {i}: True Label = {true_label}, Predicted Label = {predicted_label}")

if __name__ == "__main__":
    model = initialize_model()
    if model:
        # Example training process
        # Assuming your model is compatible with TensorFlow/Keras fit function
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout

        # Mock dataset (for actual usage, replace with your real data)
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(2, size=(100, 1))
        X_val = np.random.rand(20, 10)
        y_val = np.random.randint(2, size=(20, 1))
        X_test = np.random.rand(20, 10)  # You can replace this with your actual test set
        y_test = np.random.randint(2, size=(20, 1))

        # Example model
        keras_model = Sequential([
            Dense(64, activation='relu', input_shape=(10,), kernel_regularizer=regularizers.l2(0.01)),
            Dropout(0.5),
            Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Early stopping and learning rate scheduling callbacks
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

        # Train the model
        history = keras_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                  epochs=100, batch_size=8,
                                  callbacks=[early_stopping, reduce_lr])

        # Plot the history
        plot_model_history(history)

        # Evaluate the model on the test dataset
        evaluate_model(keras_model, X_test, y_test)

        # Save the trained model to disk
        save_model(keras_model)

        # Optionally load a saved model
        loaded_model = load_trained_model()  # You can pass the model path if necessary

        # Visualize predictions on the test data
        visualize_predictions(loaded_model, X_test, y_test)
