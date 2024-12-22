import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Load environment variables
load_dotenv()

def load_dataset_from_csv(csv_path, image_dir, target_size=(128, 128)):
    """
    Load dataset from a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing image paths and labels.
        image_dir (str): Directory where images are stored.
        target_size (tuple): Target size for resizing images (width, height).

    Returns:
        X (ndarray): Array of image data.
        y (ndarray): Array of corresponding labels.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Read CSV file
    data = pd.read_csv(csv_path)
    if 'filename' not in data.columns or 'class' not in data.columns:
        raise ValueError("CSV file must contain 'filename' and 'class' columns.")
    
    X, y = [], []

    for _, row in data.iterrows():
        img_path = os.path.join(image_dir, row['filename'])
        label = row['class']
        try:
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0  # Normalize pixel values
            X.append(img_array)
            y.append(label)
        except Exception as e:
            print(f"Skipping image {img_path}: {e}")

    return np.array(X), np.array(y)

def plot_metrics(history):
    """
    Plot training and validation accuracy and loss.

    Args:
        history: History object from training.
    """
    # Plot accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
    csv_path = os.path.join(script_dir, "valid/_annotations.csv")  # Path to the CSV file
    image_dir = os.path.join(script_dir, "valid")  # Path to the Dataset directory

    # Load dataset
    print("Loading dataset...")
    X, y = load_dataset_from_csv(csv_path, image_dir)
    print(f"Dataset loaded: {len(X)} images, {len(y)} labels.")

    # Encode labels to integers
    class_labels = sorted(list(set(y)))  # Get unique classes
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}
    y_encoded = np.array([label_to_index[label] for label in y])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Convert labels to one-hot encoding
    num_classes = len(class_labels)
    y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes)

    # Define model
    keras_model = Sequential([
        tf.keras.layers.Input(shape=X_train.shape[1:]),
        tf.keras.layers.Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    keras_model.compile(optimizer='adam', 
                        loss='categorical_crossentropy', 
                        metrics=['accuracy'])

    # Train the model
    print("Training the model...")
    history = keras_model.fit(X_train, y_train_categorical, epochs=10, batch_size=32, validation_split=0.1)

    # Plot accuracy and loss
    print("Generating accuracy and loss plots...")
    plot_metrics(history)
