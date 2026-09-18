import os
import cv2
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("=" * 65)
print("🚀 RUNNING 75:25 TRAIN-TEST SPLIT ON CLEAN MULTIMODAL DATASET")
print("=" * 65)

# ==========================================
# 1. SOIL VISION MODEL (Clean_Soil_Patches)
# ==========================================
print("\n--- [Stage 1] Soil Vision Classifier (75:25 Split) ---")

# Train on the hand-free, center-cropped patches
IMAGE_DIR = "Clean_Soil_Patches"
if not os.path.exists(IMAGE_DIR):
    IMAGE_DIR = "Soil types"
    if not os.path.exists(IMAGE_DIR) and os.path.exists("Soil types/Soil types"):
        IMAGE_DIR = "Soil types/Soil types"

classes = ["Black Soil", "Laterite Soil", "Peat Soil", "Yellow Soil", "Cinder Soil"]
img_features, img_labels = [], []

if os.path.exists(IMAGE_DIR):
    for fname in os.listdir(IMAGE_DIR):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Match class name from patch prefix or subfolder
        matched_class = None
        for c in classes:
            if fname.startswith(c.replace(" ", "_")):
                matched_class = c
                break
        
        # Fallback if checking directory hierarchy
        if matched_class is None:
            continue

        img_path = os.path.join(IMAGE_DIR, fname)
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
        img_labels.append(matched_class)

    X_img = np.array(img_features)
    y_img = np.array(img_labels)

    X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
        X_img, y_img, test_size=0.25, random_state=42, stratify=y_img
    )

    print(f"Total Pure Soil Patches : {len(X_img)}")
    print(f"Training Set (75%)      : {len(X_train_img)} images")
    print(f"Testing Set  (25%)      : {len(X_test_img)} images")

    vision_model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
    vision_model.fit(X_train_img, y_train_img)

    y_pred_img = vision_model.predict(X_test_img)
    vision_acc = accuracy_score(y_test_img, y_pred_img)
    print(f"✅ Soil Vision Test Accuracy (Unseen 25%): {vision_acc * 100:.2f}%")

    with open("soil_vision_model.pkl", "wb") as f:
        pickle.dump(vision_model, f)
    with open("test_data_vision.pkl", "wb") as f:
        pickle.dump((X_test_img, y_test_img), f)
        
# ==========================================
# 2. EXTRACT BASELINES FROM MULTIMODAL EXCEL
# ==========================================
print("\n--- [Stage 2] Indexing Baselines from soil_clean_multimodal.xlsx ---")

soil_defaults = {}
excel_file = "soil_clean_multimodal.xlsx"

if os.path.exists(excel_file):
    df_clean = pd.read_excel(excel_file)
    
    # Helper to find column name regardless of exact spelling/abbreviation
    def get_col(candidates):
        for c in df_clean.columns:
            for cand in candidates:
                if cand.lower() in c.lower():
                    return c
        return None

    col_soil = get_col(["soil class", "soil type"])
    col_tex = get_col(["texture"])
    col_n = get_col(["nitrogen", "available n", "n (kg/ha)"])
    col_p = get_col(["phosphorus", "available p", "p (kg/ha)"])
    col_k = get_col(["potassium", "available k", "k (kg/ha)"])
    col_ph = get_col(["ph"])
    col_oc = get_col(["organic carbon", "oc"])
    col_ec = get_col(["ec", "electrical cond", "salinity"])
    col_moist = get_col(["moisture"])

    for s_name, grp in df_clean.groupby(col_soil):
        soil_defaults[s_name] = {
            "texture": str(grp[col_tex].iloc[0]) if col_tex else "Loamy Sand",
            "N": round(float(grp[col_n].mean()), 1) if col_n else 65.0,
            "P": round(float(grp[col_p].mean()), 1) if col_p else 40.0,
            "K": round(float(grp[col_k].mean()), 1) if col_k else 45.0,
            "ph": round(float(grp[col_ph].mean()), 2) if col_ph else 6.8,
            "OC": round(float(grp[col_oc].mean()), 2) if col_oc else 0.55,
            "EC": round(float(grp[col_ec].mean()), 2) if col_ec else 0.35,
            "moisture": round(float(grp[col_moist].mean()), 1) if col_moist else 15.0,
            "score": 88
        }
    print(f"✅ Indexed {len(soil_defaults)} soil classes with OC, EC, and Texture directly from Excel.")
else:
    # Fallback to ICAR standards if Excel is absent
    soil_defaults = {
        "Black Soil": {"texture": "Clayey (Vertisol)", "N": 80.0, "P": 45.0, "K": 51.0, "ph": 7.35, "OC": 0.77, "EC": 0.46, "moisture": 18.4, "score": 92},
        "Laterite Soil": {"texture": "Sandy Loam (Ultisol)", "N": 40.0, "P": 24.0, "K": 34.0, "ph": 5.75, "OC": 0.37, "EC": 0.20, "moisture": 12.1, "score": 75},
        "Peat Soil": {"texture": "Organic Muck (Histosol)", "N": 95.5, "P": 30.0, "K": 20.0, "ph": 5.22, "OC": 2.71, "EC": 0.60, "moisture": 27.1, "score": 70},
        "Yellow Soil": {"texture": "Loamy Sand (Inceptisol)", "N": 50.0, "P": 35.0, "K": 40.0, "ph": 6.42, "OC": 0.44, "EC": 0.26, "moisture": 14.5, "score": 82},
        "Cinder Soil": {"texture": "Gravelly Sand (Entisol)", "N": 59.5, "P": 40.0, "K": 44.5, "ph": 6.81, "OC": 0.52, "EC": 0.34, "moisture": 9.3, "score": 85}
    }

with open("soil_chemistry_baseline.pkl", "wb") as f:
    pickle.dump({"soil_defaults": soil_defaults}, f)
print("✅ Saved chemical & physical baselines into 'soil_chemistry_baseline.pkl'")

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

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_crop, y_crop, test_size=0.25, random_state=42, stratify=y_crop
    )

    crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
    crop_model.fit(X_train_c, y_train_c)

    y_pred_c = crop_model.predict(X_test_c)
    crop_acc = accuracy_score(y_test_c, y_pred_c)
    print(f"✅ Crop Recommender Test Accuracy (Unseen 25%): {crop_acc * 100:.2f}%")

    with open("crop_recommender_model.pkl", "wb") as f:
        pickle.dump(crop_model, f)
    with open("test_data_crop.pkl", "wb") as f:
        pickle.dump((X_test_c, y_test_c), f)

print("\n" + "=" * 65)
print("🎉 TRAINING COMPLETE: Models & 25% Test Sets Exported Successfully!")
print("=" * 65)