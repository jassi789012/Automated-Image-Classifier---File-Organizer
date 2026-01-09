import os
import shutil
import cv2  # Using cv2 to match your training preprocessing
import numpy as np
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
MODEL_PATH = 'mobilenet_v2_finetuned_95acc.keras'  # Update with your actual model name
SOURCE_FOLDER = 'new_image'
IMG_SIZE = 224

# Mapping must match the order in your 'folders' list from training
# folders = ["Kartik", "Jasveer", "Manya", "jassi"]
CLASS_MAP = {
    0: 'Kartik',
    1: 'Jasveer',
    2: 'Manya',
    3: 'jassi'
}

def sort_images():
    # 1. Load Model
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Check Source Folder
    if not os.path.exists(SOURCE_FOLDER):
        print(f"Error: Folder '{SOURCE_FOLDER}' does not exist.")
        return

    # 3. Process Images
    files = os.listdir(SOURCE_FOLDER)
    valid_exts = ('.jpg', '.png', '.jpeg')
    
    count = 0
    for filename in files:
        if filename.lower().endswith(valid_exts):
            file_path = os.path.join(SOURCE_FOLDER, filename)
            
            try:
                # --- PREPROCESSING (Must match training exactly) ---
                img = cv2.imread(file_path)
                
                if img is None:
                    print(f"Skipping corrupt image: {filename}")
                    continue

                # Resize to (224, 224)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                
                # Normalize (0 to 1)
                img = img / 255.0
                
                # Expand dims to match batch shape (1, 224, 224, 3)
                img_batch = np.expand_dims(img, axis=0)
                # ---------------------------------------------------

                # Predict
                predictions = model.predict(img_batch, verbose=0)
                predicted_class = np.argmax(predictions, axis=1)[0]
                
                # Move File
                if predicted_class in CLASS_MAP:
                    target_folder = CLASS_MAP[predicted_class]
                    
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)
                        
                    dest_path = os.path.join(target_folder, filename)
                    shutil.move(file_path, dest_path)
                    
                    print(f"Moved {filename} -> {target_folder} (Class {predicted_class})")
                    count += 1
                else:
                    print(f"Warning: Unknown class {predicted_class} for {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print("---")
    print(f"Done. Sorted {count} images.")

if __name__ == "__main__":
    sort_images()