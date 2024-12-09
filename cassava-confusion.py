import os
import numpy as np
from dotenv import load_dotenv
from roboflow import Roboflow
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf

# Load environment variables
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

def plot_confusion_matrix_multiclass(model, X_test, y_test, class_labels):
    """
    Plot the confusion matrix for a multiclass classification model.

    Args:
        model: The trained model (Roboflow or Keras).
        X_test (ndarray): Test feature data.
        y_test (ndarray): True labels for the test data.
        class_labels (list): List of class labels (e.g., ["CGM", "CBSD", "CMD", "HEALTHY"]).
    """
    # Predict class probabilities and convert to predicted classes
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class indices

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')

    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    # Define class labels
    class_labels = ["CGM", "CBSD", "CMD", "HEALTHY"]

    # Example mock dataset
num_classes = len(class_labels)
X_train = np.random.rand(100, 10)  # Training features
y_train = np.random.randint(num_classes, size=(100,))  # Training labels as integer class indices
X_test = np.random.rand(20, 10)  # Test features
y_test = np.random.randint(num_classes, size=(20,))  # Test labels as integer class indices

# Mock trained model (replace with your actual Roboflow/Keras model)
keras_model = Sequential([
    tf.keras.Input(shape=(10,)),  # Explicitly define input shape
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')  # Multiclass output
])
keras_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
keras_model.fit(X_train, y_train, epochs=200, batch_size=4)

# Plot the confusion matrix
plot_confusion_matrix_multiclass(keras_model, X_test, y_test, class_labels)

