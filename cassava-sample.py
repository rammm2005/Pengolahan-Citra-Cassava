import os
import cv2
import json
import time
import numpy as np
from dotenv import load_dotenv
from roboflow import Roboflow
import matplotlib.pyplot as plt

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

def detect_disease(image_path, model):
    """
    Detects diseases in the provided image using the model and saves the annotated image.

    Args:
        image_path (str): Path to the input image.
        model: The loaded Roboflow model.
    """
    if not os.path.isfile(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image '{image_path}'.")
        return

    try:
        results = model.predict(image_path).json()
        predictions = results.get("predictions", [])
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    if predictions:
        img = draw_bounding_boxes(img, predictions)
        img = resize_image(img)

        output_folder = "public"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "output_image.jpg")
        cv2.imwrite(output_path, img)
        print(f"Detection saved as '{output_path}'")

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
    else:
        print("No predictions were found for this image.")

if __name__ == "__main__":
    model = initialize_model()
    if model:
        detect_disease("valid/10606813_jpg.rf.9099a6130c06676afc76053fde95f5e6.jpg", model)
