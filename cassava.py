import os
import requests
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Roboflow API configuration
api_url = "https://detect.roboflow.com/newcassava/3"
api_key = "SBgkktUdJ0rvjOiQPPIz"

# Path to the dataset
dataset_path = "/valid"  # Replace with your valid dataset path

# Class labels
class_labels = ["healthy", "mosaic", "necrosis"]

# Initialize ground truth and predictions
ground_truths = []
predictions = []

# Function to get predictions from Roboflow
def get_prediction(image_path):
    with open(image_path, "rb") as image_file:
        response = requests.post(
            f"{api_url}?api_key={api_key}",
            files={"file": image_file}
        )
        response_data = response.json()
        if "predictions" in response_data and response_data["predictions"]:
            # Assuming the first prediction is the most confident
            return response_data["predictions"][0]["class"]
        return None

# Walk through the dataset
for label in class_labels:
    class_dir = os.path.join(dataset_path, label)
    if not os.path.exists(class_dir):
        print(f"Directory {class_dir} not found. Skipping...")
        continue

    for image_file in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_file)
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # Skip non-image files

        # Append ground truth
        ground_truths.append(label)

        # Get prediction
        prediction = get_prediction(image_path)
        if prediction:
            predictions.append(prediction)
        else:
            predictions.append("unknown")  # Handle case where no prediction is made

# Compute confusion matrix
cm = confusion_matrix(ground_truths, predictions, labels=class_labels + ["unknown"])

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels + ["unknown"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Cassava Leaf Detection")
plt.show()
