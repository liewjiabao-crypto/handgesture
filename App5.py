import cv2
import numpy as np
import tensorflow as tf
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Prevent TensorFlow Console Spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_hand_gesture_model.h5')
IMG_SIZE = 150

GESTURE_NAMES = ['call_me', 'fingers_crossed', 'okay', 'paper', 'peace', 'rock', 'rock_on', 'scissor', 'thumbs', 'up']

class GestureApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Hand Gesture Recognition System")
        self.window.geometry("1000x700")
        self.window.configure(bg="#f0f0f0")

        self.model = None
        self.original_image = None
        self.preprocessed_image = None

        self.setup_ui()
        self.window.after(100, self.load_model)

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Error", f"Model file not found:\n{MODEL_PATH}")
            return
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.lbl_result.config(text="Model Ready. Upload an Image.", fg="blue")
        except Exception as e:
            messagebox.showerror("Model Error", str(e))

    def setup_ui(self):
        main_frame = tk.Frame(self.window, bg="#f0f0f0")
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)

        tk.Label(main_frame, text="Hand Gesture Recognition", font=("Arial", 28, "bold"), bg="#f0f0f0").pack(pady=20)

        # Image Container
        img_container = tk.Frame(main_frame, bg="#f0f0f0")
        img_container.pack(pady=20)

        # Labels for labels (Text headers)
        tk.Label(img_container, text="Original Image", bg="#f0f0f0", font=("Arial", 12, "bold")).grid(row=0, column=0)
        tk.Label(img_container, text="Processed (Skin Mask)", bg="#f0f0f0", font=("Arial", 12, "bold")).grid(row=0, column=1)

        # --- FIX: Removed fixed width/height from labels ---
        self.panel_orig = tk.Label(img_container, bg="white", relief="solid", bd=2)
        self.panel_orig.grid(row=1, column=0, padx=20, pady=10)
        # Placeholder size so the UI doesn't jump
        self.panel_orig.config(width=50, height=20) 
        self.panel_orig.pack_propagate(0)

        self.panel_proc = tk.Label(img_container, bg="white", relief="solid", bd=2)
        self.panel_proc.grid(row=1, column=1, padx=20, pady=10)
        self.panel_proc.config(width=50, height=20)
        self.panel_proc.pack_propagate(0)

        self.lbl_result = tk.Label(main_frame, text="Status: Initializing...", font=("Arial", 22), fg="blue", bg="#f0f0f0")
        self.lbl_result.pack(pady=20)

        btn_frame = tk.Frame(main_frame, bg="#f0f0f0")
        btn_frame.pack(pady=20)

        self.btn_upload = tk.Button(btn_frame, text="1. Upload Image", command=self.upload_and_preprocess, width=18, height=2, font=("Arial", 14), bg="#e0e0e0")
        self.btn_upload.grid(row=0, column=0, padx=10)

        self.btn_predict = tk.Button(btn_frame, text="2. Predict Image", command=self.predict_logic, width=18, height=2, font=("Arial", 14), bg="#e0e0e0")
        self.btn_predict.grid(row=0, column=1, padx=10)

        self.btn_quit = tk.Button(btn_frame, text="3. Quit App", command=self.close_all, width=18, height=2, font=("Arial", 14), bg="#ff9999")
        self.btn_quit.grid(row=0, column=2, padx=10)

    def upload_and_preprocess(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return

        self.original_image = cv2.imread(file_path)
        if self.original_image is None:
            messagebox.showerror("Error", "Could not load image.")
            return

        # Show Original Image
        self.display_on_label(self.original_image, self.panel_orig)

        # PREPROCESS
        img = self.original_image.copy()
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        smooth = cv2.GaussianBlur(skin_mask, (5, 5), 0)
        _, final_mask = cv2.threshold(smooth, 128, 255, cv2.THRESH_BINARY)

        self.preprocessed_image = final_mask
        self.display_on_label(self.preprocessed_image, self.panel_proc, is_gray=True)
        self.lbl_result.config(text="Image Uploaded & Preprocessed", fg="green")

    def predict_logic(self):
        if self.preprocessed_image is None or self.model is None:
            messagebox.showwarning("Warning", "Upload an image and ensure model is loaded!")
            return

        inp = cv2.resize(self.preprocessed_image, (IMG_SIZE, IMG_SIZE))
        inp = inp.astype('float32') / 255.0
        inp = np.expand_dims(inp, axis=(0, -1))

        pred = self.model.predict(inp, verbose=0)
        idx = np.argmax(pred)
        conf = pred[0][idx]
        gesture = GESTURE_NAMES[idx].upper()

        self.lbl_result.config(text=f"Prediction: {gesture} ({conf*100:.1f}%)", fg="green")

    def display_on_label(self, cv_img, label, is_gray=False):
        try:
            # 1. Resize logic
            h, w = cv_img.shape[:2]
            scaling = 300 / max(h, w)
            new_w, new_h = int(w * scaling), int(h * scaling)
            cv_img_resized = cv2.resize(cv_img, (new_w, new_h))

            # 2. Convert to PIL
            if is_gray:
                img_pil = Image.fromarray(cv_img_resized)
            else:
                img_rgb = cv2.cvtColor(cv_img_resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)

            # 3. Create PhotoImage and EXPLICITLY set the master
            # This 'master=self.window' is the critical fix for "pyimage" errors
            img_tk = ImageTk.PhotoImage(img_pil, master=self.window)

            # 4. Update label
            label.configure(image=img_tk, width=new_w, height=new_h)
            
            # 5. Keep reference (Crucial to prevent garbage collection)
            label.image = img_tk

        except Exception as e:
            print("Display Error:", e)

    def close_all(self):
        self.window.destroy()
        os._exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    # If using Spyder/IPython, this helps clear old instances
    try:
        root.destroy() 
    except:
        pass

    print("Starting Application...")
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()