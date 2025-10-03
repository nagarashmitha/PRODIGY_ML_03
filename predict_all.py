from joblib import load
import cv2
import numpy as np
import os
import csv

# -------- Load scaler and SVM model -----------
scaler = load("scaler.pkl")
model = load("svm_cat_dog.pkl")

print("Scaler and SVM model loaded successfully!")

# -------- Determine expected input size from scaler --------
n_features = scaler.mean_.shape[0]  # e.g., 34020
channels = 3
pixels = n_features // channels  # 11340

def find_image_size(pixels):
    factors = [(i, pixels // i) for i in range(50, 200) if pixels % i == 0]
    h, w = min(factors, key=lambda x: abs(x[0] - x[1]))  # closest to square
    return h, w

height, width = find_image_size(pixels)
print(f"Detected training size: {width}x{height} (total features {n_features})")

# -------- Define dataset folder -----------
dataset_folder = "dataset/train"

# -------- CSV setup -----------
csv_file = "predictions.csv"
correct, total = 0, 0  # for accuracy

with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "true_label", "predicted_label"])  # header row

    # -------- Loop through subfolders (cats, dogs) -----------
    for label_name in os.listdir(dataset_folder):
        subfolder = os.path.join(dataset_folder, label_name)
        if not os.path.isdir(subfolder):
            continue

        # Normalize true label (remove plural if needed)
        if "cat" in label_name.lower():
            true_label = "Cat"
        elif "dog" in label_name.lower():
            true_label = "Dog"
        else:
            continue  # skip unknown folders

        print(f"\nProcessing images in: {subfolder}")

        for img_file in os.listdir(subfolder):
            img_path = os.path.join(subfolder, img_file)
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            # Load and preprocess image
            image = cv2.imread(img_path)
            image = cv2.resize(image, (width, height))
            image = image.flatten()
            image_scaled = scaler.transform([image])

            # Predict
            prediction = model.predict(image_scaled)
            label_map = {0: "Cat", 1: "Dog"}  # adjust if needed
            predicted_label = label_map.get(prediction[0], str(prediction[0]))

            print(f"{img_file}: True -> {true_label}, Predicted -> {predicted_label}")

            # Save to CSV
            writer.writerow([img_file, true_label, predicted_label])

            # Track accuracy
            total += 1
            if predicted_label == true_label:
                correct += 1

# -------- Show accuracy --------
if total > 0:
    accuracy = correct / total * 100
    print(f"\n✅ Predictions saved to {csv_file}")
    print(f"📊 Accuracy: {correct}/{total} = {accuracy:.2f}%")
else:
    print("\n⚠️ No images found to process!")










