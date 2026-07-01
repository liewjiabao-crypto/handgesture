import os, sys, random, json, csv, gc, time
from pathlib import Path
from collections import Counter

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- Reproducibility ---
SEED = 3
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- GPU safety ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# === Configuration ===
BASE_DIRS = [
    r'C:\Users\user\Documents\BDP 20262027\HandGesture\images',
    r'C:\Users\user\Documents\BDP 20262027\DATASET\Train_Data'
]

OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_WIDTH, NEW_HEIGHT = 150, 150

FILES_NAMES = [
    'call_me', 'fingers_crossed', 'okay', 'paper', 'peace',
    'rock', 'rock_on', 'scissor', 'thumbs', 'up'
]
CLASS_MAP = {name: idx for idx, name in enumerate(FILES_NAMES)}
ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
NUM_CLASSES = len(FILES_NAMES)

BATCH_SIZES = [8, 16, 32, 64] 
EPOCHS_LIST = [5, 10, 15, 20] 

# === Helpers ===
def load_dataset(base_dirs, gesture_names, width, height):
    all_image_paths = []
    valid_exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    for dir_path in base_dirs:
        base = Path(dir_path)
        if not base.exists(): continue
        dir_count = 0
        for g in gesture_names:
            gpath = base / g
            if not gpath.exists(): continue
            for ext in valid_exts:
                for p in gpath.rglob(ext):
                    all_image_paths.append((str(p), CLASS_MAP[g]))
                    dir_count += 1
        print(f"✅ Found {dir_count} images in {base.name}")

    X, y = [], []
    for img_path, label in tqdm(all_image_paths, desc="Loading"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        X.append(img)
        y.append(label)

    X = np.stack(X, axis=0).astype('float32') / 255.0
    X = X.reshape(-1, height, width, 1)
    y = np.array(y, dtype=np.int64)
    return X, y

# Augmentation
def apply_custom_augmentation(X, y):
    n = len(X)
    idx1, idx2 = int(n * 0.33), int(n * 0.66) 
    X_horiz, y_horiz = np.flip(X[:idx1], axis=2), y[:idx1]
    X_vert, y_vert = np.flip(X[idx1:idx2], axis=1), y[idx1:idx2]
    X_orig, y_orig = X[idx2:], y[idx2:]
    X_final = np.concatenate([X_horiz, X_vert, X_orig], axis=0)
    y_final = np.concatenate([y_horiz, y_vert, y_orig], axis=0)
    indices = np.arange(len(X_final))
    np.random.shuffle(indices)
    return X_final[indices], y_final[indices]

# Cnn model architecture
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (5, 5), activation='relu', padding='valid', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='valid'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='valid'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Top 3 misclassified
def analyze_top_errors(X_test, y_test, y_pred, class_names):
    cm = confusion_matrix(y_test, y_pred)
    cm_errors = cm.copy()
    np.fill_diagonal(cm_errors, 0) 
    flat_indices = np.argsort(cm_errors.ravel())[-3:][::-1]
    top_3_indices = [np.unravel_index(i, cm.shape) for i in flat_indices]
    
    plt.figure(figsize=(15, 5))
    for i, (true_idx, pred_idx) in enumerate(top_3_indices):
        count = cm[true_idx, pred_idx]
        error_indices = np.where((y_test == true_idx) & (y_pred == pred_idx))[0]
        plt.subplot(1, 3, i + 1)
        if len(error_indices) > 0:
            plt.imshow(X_test[random.choice(error_indices)].reshape(NEW_HEIGHT, NEW_WIDTH), cmap='gray')
            plt.title(f"TRUE: {class_names[true_idx]}\nPRED: {class_names[pred_idx]}\n({count} cases)", color='red')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plots Training/Validation Accuracy and Loss for the best model
def plot_best_learning_curves(history, label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1.plot(history['accuracy'], label='Train Accuracy', marker='o')
    ax1.plot(history['val_accuracy'], label='Val Accuracy', marker='s')
    ax1.set_title(f'Best Model Accuracy ({label})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history['loss'], label='Train Loss', color='red', marker='o')
    ax2.plot(history['val_loss'], label='Val Loss', color='darkred', marker='s')
    ax2.set_title(f'Best Model Loss ({label})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Training time vs test acc
def plot_runtime_performance(results):
    labels = [r['label'] for r in results]
    runtimes = [r['runtime'] for r in results]
    accuracies = [r['test_accuracy'] for r in results]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Experimental Run (BS:EP)')
    ax1.set_ylabel('Test Accuracy', color=color)
    ax1.bar(labels, accuracies, color=color, alpha=0.4, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(min(accuracies)-0.05, 1.0)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Runtime (seconds)', color=color)
    ax2.plot(labels, runtimes, color=color, marker='D', linewidth=2, label='Runtime')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Performance vs. Training Time for All Configurations')
    fig.tight_layout()
    plt.show()

# === Main ===
def main():
    X_raw, y_raw = load_dataset(BASE_DIRS, FILES_NAMES, NEW_WIDTH, NEW_HEIGHT)

    # 60/20/20 Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=SEED, stratify=y_raw
    )
    X_train_raw, X_val, y_train_raw, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp
    )

    X_train, y_train = apply_custom_augmentation(X_train_raw, y_train_raw)

    results = []
    best_run_info = None
    best_model_obj = None

    for bs in BATCH_SIZES:
        for ep in EPOCHS_LIST:
            tf.keras.backend.clear_session()
            gc.collect()

            print(f"\n=== Training Configuration: BS={bs}, Epochs={ep} ===")
            model = build_model(input_shape=(NEW_HEIGHT, NEW_WIDTH, 1), num_classes=NUM_CLASSES)
            
            start_time = time.time()
            history = model.fit(
                X_train, y_train,
                epochs=ep, batch_size=bs,
                validation_data=(X_val, y_val), verbose=1
            )
            runtime = time.time() - start_time
            
            _, test_acc = model.evaluate(X_test, y_test, verbose=0)
            
            run_data = {
                'label': f"{bs}:{ep}",
                'batch_size': bs, 'epochs': ep,
                'train_acc': history.history['accuracy'][-1],
                'val_acc': history.history['val_accuracy'][-1],
                'test_accuracy': test_acc,
                'runtime': runtime,
                'history': history.history
            }
            results.append(run_data)

            if best_run_info is None or run_data['test_accuracy'] > best_run_info['test_accuracy']:
                best_run_info = run_data
                best_model_obj = model

    # 1. Performance Table
    print("\nTable 4.1: Summary of Experimental Runs Performance")
    print("-" * 100)
    print(f"{'Run (BS:EP)':<15} {'Runtime (s)':<15} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}")
    for r in results:
        print(f"{r['label']:<15} {r['runtime']:<15.2f} {r['train_acc']:<12.4f} {r['val_acc']:<12.4f} {r['test_accuracy']:<12.4f}")
    
    # 2. Plot: Best Learning Curves (Loss/Accuracy)
    plot_best_learning_curves(best_run_info['history'], best_run_info['label'])

    # 3. Plot: Time vs Performance for all runs
    plot_runtime_performance(results)

    if best_model_obj:
        y_pred = np.argmax(best_model_obj.predict(X_test), axis=1)
        
        # 4. Top 3 Mistakes
        analyze_top_errors(X_test, y_test, y_pred, FILES_NAMES)

        # 5. Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
                    xticklabels=FILES_NAMES, yticklabels=FILES_NAMES)
        plt.title(f'Confusion Matrix (Best Run: {best_run_info["label"]})')
        plt.show()

        # 6. Save Best Model
        save_path = OUT_DIR / "best_hand_gesture_model.h5"
        best_model_obj.save(save_path)
        print(f"\n🚀 Best Model Saved to: {save_path}")

if __name__ == "__main__":
    main()