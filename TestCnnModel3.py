import os
import cv2
import time
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURATION (REPLACE THESE VALUES)
# ==========================================
MODEL_PATH = 'models/best_hand_gesture_model.h5'  # <--- REPLACE THIS
DATASET_PATH = r'C:\Users\user\Documents\BDP 20262027\DATASET\DataProcessedNew'   # Your processed images folder

# IMPORTANT: Change this to match the image size your model was trained on!
# For example, if you trained on 224x224 images, set this to (224, 224)
IMG_SIZE = (150, 150)
# ==========================================
# 2. LOAD DATA & SHUFFLE
# ==========================================
print("Loading and shuffling dataset...")

# Get all the folder names (which act as your class labels)
class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
class_map = {name: idx for idx, name in enumerate(class_names)}

dataset = []

# Read all images
for class_name in class_names:
    folder_path = os.path.join(DATASET_PATH, class_name)
    label_idx = class_map[class_name]
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            dataset.append((img_path, label_idx))

# SHUFFLE the dataset completely
random.shuffle(dataset)

# Prepare arrays for the model
X_test = []
y_true = []

for img_path, label in dataset:
    # 1. Read the image directly in Grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. Resize to match model input
    img = cv2.resize(img, IMG_SIZE)
    
    # 3. Add the channel dimension back so it becomes (128, 128, 1) instead of just (128, 128)
    img = np.expand_dims(img, axis=-1)
    
    # 4. Normalize
    img = img / 255.0
    
    X_test.append(img)
    y_true.append(label)

X_test = np.array(X_test)
y_true = np.array(y_true)

print(f"Total images loaded for testing: {len(X_test)}")

# ==========================================
# 3. LOAD MODEL & EVALUATE
# ==========================================
print(f"Loading model from: {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Starting evaluation...")
start_time = time.time()

# Make predictions on the entire shuffled dataset
predictions = model.predict(X_test)

end_time = time.time()
evaluation_time = end_time - start_time

# Convert probabilities to class indices
y_pred = np.argmax(predictions, axis=1)

# ==========================================
# 4. CALCULATE METRICS
# ==========================================
accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

print("-" * 30)
print("EVALUATION RESULTS")
print("-" * 30)
print(f"Total Time Taken: {evaluation_time:.4f} seconds")
print(f"Average Time per Image: {(evaluation_time / len(X_test)):.6f} seconds")
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
print("-" * 30)

# ==========================================
# 5. PLOT CONFUSION MATRIX
# ==========================================
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)

plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('True Gesture', fontsize=12)
plt.xlabel('Predicted Gesture', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as an image and show it
plt.savefig('confusion_matrix_results.png')
print("Saved confusion matrix as 'confusion_matrix_results.png'")
plt.show()