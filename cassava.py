import torch
import cv2
import numpy as np
from roboflow import Roboflow
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import json
import time

# Load environment variables from .env file
load_dotenv()

def initialize_model():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("API key for Roboflow is not set. Please set ROBOFLOW_API_KEY as an environment variable.")
    # Initialize Roboflow and load the project model
    rf = Roboflow(api_key=api_key)
    try:
        project = rf.workspace().project("newcassava")
        model = project.version("3").model
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def detect_disease(image_path, model):
    # Check if image file exists
    if not os.path.isfile(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image '{image_path}'.")
        return

    # Run inference with the model
    try:
        results = model.predict(image_path).json()  # parse JSON results
        print("Full Prediction JSON Response:\n", json.dumps(results, indent=4))  # Pretty-print JSON response
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # Variable to accumulate confidence scores
    total_confidence = 0
    num_predictions = 0

    # Create output folder if it doesn't exist
    output_folder = "public"
    os.makedirs(output_folder, exist_ok=True)

    # Draw bounding boxes on detected diseases if predictions are available
    if 'predictions' in results and results['predictions']:
        for prediction in results['predictions']:
            print("Individual Prediction Entry:\n", json.dumps(prediction, indent=4))  # Pretty-print each prediction entry
            
            # Verify the correct key for class
            disease_name = prediction.get('class') or prediction.get('label') or "Unknown"
            confidence = prediction.get('confidence', 0)

            # Calculate bounding box coordinates
            x0 = int(prediction['x'] - prediction['width'] / 2)
            y0 = int(prediction['y'] - prediction['height'] / 2)
            x1 = int(prediction['x'] + prediction['width'] / 2)
            y1 = int(prediction['y'] + prediction['height'] / 2)

            # Draw rectangle on the image
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 10)

            # Label with disease name and confidence
            label = f"Disease: {disease_name} ({confidence:.2f})"
            total_confidence += confidence
            num_predictions += 1

            # Put the label on the image
            cv2.putText(img, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Calculate average confidence
        avg_confidence = total_confidence / num_predictions if num_predictions > 0 else 0.0
        accuracy_label = f"Prediction Accuracy: {avg_confidence * 100:.2f}%, (Disease Name: {disease_name})"
        cv2.putText(img, accuracy_label, (50, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

        # Resize image to max width 800 pixels if necessary
        max_width = 800
        height, width = img.shape[:2]
        if width > max_width:
            scaling_factor = max_width / float(width)
            new_dim = (max_width, int(height * scaling_factor))
            img = cv2.resize(img, new_dim)

        # Save the image with detections
        # generate_name = time.strftime("%Y_%m_%d_%p")
        output_path = os.path.join(output_folder, "output_image.jpg")
        cv2.imwrite(output_path, img)
        print(f"Detection saved as '{output_path}'")

        # Display image with Matplotlib
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        print("No predictions were found for this image.")

# Initialize model and run detection
model = initialize_model()
if model:
    # Example usage
    detect_disease("image/IMG_7811.JPG", model)