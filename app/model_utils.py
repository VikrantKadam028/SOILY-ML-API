# model_utils.py - FIXED VERSION
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
USER_DATASET_PATH = os.path.join(BASE_DIR, "data", "predicted_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_bundle.pkl")
MODEL_STATS_PATH = os.path.join(BASE_DIR, "models", "model_stats.pkl")

FEATURES_CATEGORICAL = ["District_Name", "Soil_color"]
FEATURES_NUMERIC = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Rainfall", "Temperature"]
ALL_FEATURES = FEATURES_CATEGORICAL + FEATURES_NUMERIC
LABELS = ["Crop", "Fertilizer"]

model_stats = {
    'crop_accuracy': 0.0,
    'fertilizer_accuracy': 0.0,
    'training_date': None,
    'dataset_size': 0,
}

def load_main_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"dataset.csv missing at: {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    
    # FIX: Validate and clean data
    required = ALL_FEATURES + LABELS
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in dataset.csv")
    
    # Remove rows with missing critical data
    df = df.dropna(subset=ALL_FEATURES + LABELS)
    
    
    
    return df

# def load_user_generated_dataset():
#     if not os.path.exists(USER_DATASET_PATH):
#         return pd.DataFrame(columns=ALL_FEATURES + LABELS)
    
#     if os.path.getsize(USER_DATASET_PATH) == 0:
#         return pd.DataFrame(columns=ALL_FEATURES + LABELS)
    
#     try:
#         df = pd.read_csv(USER_DATASET_PATH)
#         df = df.dropna(subset=ALL_FEATURES + LABELS)
        
#         # Normalize nitrogen if needed
#         if len(df) > 0 and df['Nitrogen'].max() > 10:
#             df['Nitrogen'] = df['Nitrogen'] / 100.0
        
#         return df
#     except Exception:
#         return pd.DataFrame(columns=ALL_FEATURES + LABELS)

def save_user_prediction(row):
    df_new = pd.DataFrame([row])
    
    if not os.path.exists(USER_DATASET_PATH) or os.path.getsize(USER_DATASET_PATH) == 0:
        df_new.to_csv(USER_DATASET_PATH, index=False)
    else:
        df_new.to_csv(USER_DATASET_PATH, mode='a', header=False, index=False)

def build_preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CATEGORICAL),
        ("num", StandardScaler(), FEATURES_NUMERIC)
    ])
def train_models(df):
    """Train models with proper data validation + TRUE BALANCED DATASET"""
    print(f"\n{'='*60}")
    print(f"🔄 TRAINING MODELS (Oversampled + Balanced)")
    print(f"{'='*60}")
    
    if len(df) < 10:
        raise ValueError("Insufficient data for training.")

    print(f"📊 Initial dataset: {len(df)} rows")

    # 1️⃣ CLEAN DATASET
    df_clean = df.dropna(subset=ALL_FEATURES + LABELS).copy()
    print(f"✅ After cleaning: {len(df_clean)} rows")

    if len(df_clean) < 10:
        raise ValueError("Insufficient clean data.")

    # 2️⃣ OVERSAMPLE MINORITY CROPS (THE REAL FIX)
    print("\n🔧 Oversampling minority crops to 250 samples each...")

    oversampled_groups = []
    TARGET = 250  # Target rows per crop

    for crop, group in df_clean.groupby("Crop"):
        count = len(group)

        if count < TARGET:
            reps = int(TARGET / count) + 1
            new_group = pd.concat([group] * reps, ignore_index=True).head(TARGET)
        else:
            new_group = group.sample(TARGET, random_state=42)

        oversampled_groups.append(new_group)

    df_balanced = pd.concat(oversampled_groups, ignore_index=True)

    print(f"📊 After oversampling: {len(df_balanced)} rows")
    print("🌾 New distribution:", df_balanced["Crop"].value_counts().to_dict())

    # 3️⃣ BUILD CROP-FERTILIZER MAP
    print("\n📋 Building crop-fertilizer mapping...")
    
    crop_fert_map = {}
    for crop in df_balanced["Crop"].unique():
        sub = df_balanced[df_balanced["Crop"] == crop]
        fert_counts = sub["Fertilizer"].value_counts()

        if len(fert_counts) > 0:
            crop_fert_map[crop] = {
                "primary": fert_counts.index[0],
                "all": fert_counts.to_dict(),
            }

    print(f"✅ Mapped {len(crop_fert_map)} crops to fertilizers")

    # 4️⃣ TRAINING DATA
    X = df_balanced[ALL_FEATURES]
    y_crop = df_balanced["Crop"]
    y_fert = df_balanced["Fertilizer"]

    # SPLIT
    X_train, X_test, y_crop_train, y_crop_test = train_test_split(
        X, y_crop, test_size=0.2, random_state=42, stratify=y_crop
    )

    _, X_test_fert, y_fert_train, y_fert_test = train_test_split(
        X, y_fert, test_size=0.2, random_state=42, stratify=y_fert
    )

    # 5️⃣ MODELS
    pre = build_preprocessor()

    crop_model = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=250,
            max_depth=20,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    fert_model = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=250,
            max_depth=20,
            random_state=43,
            class_weight="balanced"
        ))
    ])

    print("\n🤖 Training crop model...")
    crop_model.fit(X_train, y_crop_train)

    print("🤖 Training fertilizer model...")
    fert_model.fit(X_train, y_fert_train)

    # 6️⃣ EVALUATE
    crop_acc = accuracy_score(y_crop_test, crop_model.predict(X_test))
    fert_acc = accuracy_score(y_fert_test, fert_model.predict(X_test_fert))

    crop_cv = cross_val_score(crop_model, X, y_crop, cv=5)
    fert_cv = cross_val_score(fert_model, X, y_fert, cv=5)

    global model_stats
    model_stats = {
        "crop_accuracy": float(crop_acc),
        "fertilizer_accuracy": float(fert_acc),
        "crop_cv_mean": float(crop_cv.mean()),
        "crop_cv_std": float(crop_cv.std()),
        "fert_cv_mean": float(fert_cv.mean()),
        "fert_cv_std": float(fert_cv.std()),
        "dataset_size": len(df_balanced),
        "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    bundle = {
        "crop_model": crop_model,
        "fert_model": fert_model,
        "training_data": df_balanced.copy(),
        "crop_fert_map": crop_fert_map,
        "stats": model_stats,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    joblib.dump(model_stats, MODEL_STATS_PATH)

    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE (WITH OVERSAMPLING)!")
    print("📊 NEW Balanced Crop Counts:")
    print(df_balanced["Crop"].value_counts())
    print(f"{'='*60}\n")

    return bundle



def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    
    bundle = joblib.load(MODEL_PATH)
    
    if os.path.exists(MODEL_STATS_PATH):
        try:
            stats = joblib.load(MODEL_STATS_PATH)
            bundle['stats'] = stats
        except:
            pass
    
    return bundle

def get_model_statistics():
    bundle = load_model()
    if bundle and 'stats' in bundle:
        return bundle['stats']
    return model_stats

def find_best_fertilizer_for_crop(crop_name, input_row, training_data, crop_fert_map):
    """
    FIX: Improved fertilizer matching with fallback to crop-fertilizer map
    """
    try:
        # Strategy 1: Use pre-computed crop-fertilizer mapping
        if crop_name in crop_fert_map:
            primary_fert = crop_fert_map[crop_name]['primary']
            print(f"   💊 {crop_name} → {primary_fert} (from map)")
            return {
                "fertilizer": primary_fert,
                "confidence": 0.90
            }
        
        # Strategy 2: Find similar records in training data
        crop_records = training_data[training_data['Crop'] == crop_name].copy()
        
        if len(crop_records) == 0:
            print(f"   ⚠️  No training data for {crop_name}, using default")
            return {"fertilizer": "NPK 10-26-26", "confidence": 0.60}
        
        # Calculate similarity scores
        similarities = []
        for idx, record in crop_records.iterrows():
            score = 100.0
            score -= abs(input_row['Nitrogen'] - record['Nitrogen']) * 50
            score -= abs(input_row['Phosphorus'] - record['Phosphorus']) * 0.4
            score -= abs(input_row['Potassium'] - record['Potassium']) * 0.15
            score -= abs(input_row['pH'] - record['pH']) * 8
            score -= abs(input_row['Rainfall'] - record['Rainfall']) * 0.025
            score -= abs(input_row['Temperature'] - record['Temperature']) * 1.5
            
            similarities.append({
                'fertilizer': record['Fertilizer'],
                'score': max(0, min(100, score))
            })
        
        # Group by fertilizer
        fert_scores = {}
        for item in similarities:
            fert = item['fertilizer']
            if fert not in fert_scores:
                fert_scores[fert] = []
            fert_scores[fert].append(item['score'])
        
        # Get best
        best_fert = max(fert_scores.items(), key=lambda x: sum(x[1])/len(x[1]))
        confidence = (sum(best_fert[1]) / len(best_fert[1])) / 100.0
        
        print(f"   💊 {crop_name} → {best_fert[0]} ({confidence*100:.1f}%)")
        
        return {
            "fertilizer": best_fert[0],
            "confidence": round(confidence, 4)
        }
        
    except Exception as e:
        print(f"   ⚠️  Error matching fertilizer for {crop_name}: {e}")
        return {"fertilizer": "NPK 10-26-26", "confidence": 0.65}

def predict(bundle, row):
    """Make prediction with FIXED fertilizer matching"""
    df = pd.DataFrame([row])
    
    crop_model = bundle["crop_model"]
    fert_model = bundle["fert_model"]
    training_data = bundle.get("training_data")
    crop_fert_map = bundle.get("crop_fert_map", {})  # FIX: Get mapping
    
    # Get predictions
    crop_pred = crop_model.predict(df)[0]
    fert_pred = fert_model.predict(df)[0]
    
    # Get probabilities
    crop_proba = crop_model.predict_proba(df)[0]
    fert_proba = fert_model.predict_proba(df)[0]
    
    crop_classes = crop_model.classes_
    fert_classes = fert_model.classes_
    
    # Sort crops by probability
    crop_scores = [(crop, float(prob)) for crop, prob in zip(crop_classes, crop_proba)]
    crop_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Sort fertilizers
    fert_scores = [(fert, float(prob)) for fert, prob in zip(fert_classes, fert_proba)]
    fert_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🌾 Top 5 Crop Predictions:")
    
    # Build crop list with proper fertilizers
    top_crops = []
    for i, (crop_name, crop_prob) in enumerate(crop_scores[:5]):
        # FIX: Use improved fertilizer matching
        if training_data is not None:
            fert_info = find_best_fertilizer_for_crop(
                crop_name, 
                row, 
                training_data,
                crop_fert_map
            )
            fertilizer = fert_info['fertilizer']
            fert_conf = fert_info['confidence']
        else:
            fertilizer = fert_pred
            fert_conf = 0.70
        
        print(f"   {i+1}. {crop_name}: {crop_prob*100:.2f}% → {fertilizer}")
        
        top_crops.append({
            "crop": str(crop_name),
            "label": str(crop_name),
            "fertilizer": fertilizer,
            "fertilizer_confidence": fert_conf,
            "score": round(crop_prob, 4),
            "prob": round(crop_prob, 4),
            "percentage": round(crop_prob * 100, 2)
        })
    
    # Build fertilizer list
    top_fertilizers = []
    for fert, score in fert_scores[:5]:
        top_fertilizers.append({
            "fertilizer": str(fert),
            "label": str(fert),
            "score": round(score, 4),
            "prob": round(score, 4),
            "percentage": round(score * 100, 2)
        })
    
    primary_crop = top_crops[0]
    
    return {
        "crop": {
            "label": primary_crop["crop"],
            "confidence": primary_crop["prob"],
            "prob": primary_crop["prob"],
            "percentage": primary_crop["percentage"],
        },
        "fertilizer": {
            "label": primary_crop["fertilizer"],
            "confidence": primary_crop["fertilizer_confidence"],
            "prob": primary_crop["fertilizer_confidence"],
            "percentage": round(primary_crop["fertilizer_confidence"] * 100, 2),
        },
        "all_crops": top_crops,
        "all_fertilizers": top_fertilizers,
        "input_data": row
    }

def validate_input_data(data):
    """Validate input data"""
    required_soil = ["District_Name", "Soil_color", "Nitrogen", "Phosphorus", "Potassium", "pH"]
    required_climate = ["Rainfall", "Temperature"]
    
    soil = data.get("soil", {})
    climate = data.get("climate", {})
    
    for field in required_soil:
        if field not in soil or soil[field] is None:
            return False, f"Missing soil field: {field}"
    
    for field in required_climate:
        if field not in climate or climate[field] is None:
            return False, f"Missing climate field: {field}"
    
    try:
        if not (0 <= float(soil["pH"]) <= 14):
            return False, "pH must be between 0 and 14"
        if float(soil["Nitrogen"]) < 0 or float(soil["Phosphorus"]) < 0 or float(soil["Potassium"]) < 0:
            return False, "Nutrient values must be positive"
        if float(climate["Rainfall"]) < 0:
            return False, "Rainfall must be positive"
        if not (-50 <= float(climate["Temperature"]) <= 50):
            return False, "Temperature must be between -50°C and 50°C"
    except ValueError:
        return False, "Invalid numeric values"
    
    return True, "Valid data"