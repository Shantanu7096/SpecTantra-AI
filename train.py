import os
import cv2
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("=" * 65)
print("🚀 RUNNING 75:25 TRAIN-TEST SPLIT & EVALUATION PIPELINE")
print("=" * 65)

# ==========================================
# 1. SOIL VISION MODEL (75% Train : 25% Test)
# ==========================================
print("\n--- [Stage 1] Soil Vision Classifier (75:25 Split) ---")

IMAGE_DIR = "Soil types"
if not os.path.exists(IMAGE_DIR) and os.path.exists("Soil types/Soil types"):
    IMAGE_DIR = "Soil types/Soil types"

img_features, img_labels = [], []

if os.path.exists(IMAGE_DIR):
    for class_name in os.listdir(IMAGE_DIR):
        class_folder = os.path.join(IMAGE_DIR, class_name)
        if not os.path.isdir(class_folder):
            continue

        for img_name in os.listdir(class_folder):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(class_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (128, 128))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) / 255.0

            r_m, g_m, b_m = np.mean(img_rgb[:, :, 0]), np.mean(img_rgb[:, :, 1]), np.mean(img_rgb[:, :, 2])
            r_s, g_s, b_s = np.std(img_rgb[:, :, 0]), np.std(img_rgb[:, :, 1]), np.std(img_rgb[:, :, 2])
            h_m, s_m, v_m = np.mean(hsv[:, :, 0]), np.mean(hsv[:, :, 1]), np.mean(hsv[:, :, 2])

            img_features.append([r_m, g_m, b_m, r_s, g_s, b_s, h_m, s_m, v_m])
            img_labels.append(class_name)

    X_img = np.array(img_features)
    y_img = np.array(img_labels)

    # 75% Training, 25% Testing
    X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
        X_img, y_img, test_size=0.25, random_state=42, stratify=y_img
    )

    print(f"Total Images : {len(X_img)}")
    print(f"Training Set (75%): {len(X_train_img)} images")
    print(f"Testing Set  (25%): {len(X_test_img)} images")

    vision_model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
    vision_model.fit(X_train_img, y_train_img)

    # Measure accuracy on the 25% holdout set
    y_pred_img = vision_model.predict(X_test_img)
    vision_acc = accuracy_score(y_test_img, y_pred_img)
    print(f"✅ Soil Vision Test Accuracy (Unseen 25%): {vision_acc * 100:.2f}%")

    with open("soil_vision_model.pkl", "wb") as f:
        pickle.dump(vision_model, f)
    # Save the 25% test partition for test.py verification
    with open("test_data_vision.pkl", "wb") as f:
        pickle.dump((X_test_img, y_test_img), f)

# ==========================================
# 2. REGIONAL CHEMISTRY DATA (Soil data.csv)
# ==========================================
print("\n--- [Stage 2] Regional Indian Soil Baselines ---")
soil_defaults = {
    "Black Soil": {"N": 80.0, "P": 45.0, "K": 50.0, "ph": 7.3, "score": 90},
    "Laterite Soil": {"N": 40.0, "P": 25.0, "K": 35.0, "ph": 5.8, "score": 75},
    "Peat Soil": {"N": 95.0, "P": 30.0, "K": 20.0, "ph": 5.2, "score": 70},
    "Yellow Soil": {"N": 50.0, "P": 35.0, "K": 40.0, "ph": 6.4, "score": 82},
    "Cinder Soil": {"N": 60.0, "P": 40.0, "K": 45.0, "ph": 6.8, "score": 85}
}

district_benchmarks = {}
if os.path.exists("Soil data.csv"):
    df_soil = pd.read_csv("Soil data.csv")
    district_benchmarks = df_soil.groupby("District")[
        ['Nitrogen Value', 'Phosphorous value', 'Potassium value', 'pH']
    ].mean().to_dict('index')
    print(f"✅ Indexed {len(df_soil)} soil chemistry records across {len(district_benchmarks)} districts.")

with open("soil_chemistry_baseline.pkl", "wb") as f:
    pickle.dump({"soil_defaults": soil_defaults, "district_benchmarks": district_benchmarks}, f)

# ==========================================
# 3. CROP RECOMMENDER (75% Train : 25% Test)
# ==========================================
print("\n--- [Stage 3] Crop Recommender (75:25 Split) ---")
if os.path.exists("Crop_recommendation.csv"):
    crop_df = pd.read_csv("Crop_recommendation.csv")
    crop_df.columns = [c.strip().lower() for c in crop_df.columns]

    feature_cols = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
    X_crop = crop_df[feature_cols]
    y_crop = crop_df['label']

    # 75% Training (1,650 samples), 25% Testing (550 samples)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_crop, y_crop, test_size=0.25, random_state=42, stratify=y_crop
    )

    print(f"Total Records: {len(X_crop)}")
    print(f"Training Set (75%): {len(X_train_c)} rows")
    print(f"Testing Set  (25%): {len(X_test_c)} rows")

    crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
    crop_model.fit(X_train_c, y_train_c)

    # Evaluate on the 25% holdout set
    y_pred_c = crop_model.predict(X_test_c)
    crop_acc = accuracy_score(y_test_c, y_pred_c)
    print(f"✅ Crop Recommender Test Accuracy (Unseen 25%): {crop_acc * 100:.2f}%")

    with open("crop_recommender_model.pkl", "wb") as f:
        pickle.dump(crop_model, f)
    # Save the 25% test partition for test.py verification
    with open("test_data_crop.pkl", "wb") as f:
        pickle.dump((X_test_c, y_test_c), f)

print("\n" + "=" * 65)
print("🎉 TRAINING COMPLETE: Models & 25% Test Sets Exported Successfully!")
print("=" * 65)