from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime
import traceback
import os
from model_utils import (
    load_main_dataset,
    save_user_prediction,
    train_models,
    load_model,
    predict,
    get_model_statistics,
    validate_input_data
)

app = Flask(__name__)
CORS(app)

# Global variables
main_df = None
merged_df = None
bundle = None


def initialize_system():
    """Initialize datasets and ML models"""
    global main_df, merged_df, bundle

    try:
        print("🔄 Initializing ML System...")

        # Load main dataset
        main_df = load_main_dataset()
        merged_df = main_df.copy()

        print(f"📊 Dataset loaded successfully: {len(main_df)} samples")

        # Load or train model
        bundle = load_model()

        if bundle is None:
            print("🤖 No trained model found. Training new model...")
            bundle = train_models(merged_df)
        else:
            print("✅ Pre-trained model loaded successfully")

        print("🚀 ML System initialized successfully!")

    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        traceback.print_exc()


# Initialize system on startup
initialize_system()


# -------------------- API ENDPOINTS --------------------

@app.route("/predict", methods=["POST"])
def do_predict():
    try:
        data = request.json

        is_valid, message = validate_input_data(data)
        if not is_valid:
            return jsonify({
                "success": False,
                "error": message
            }), 400

        soil = data["soil"]
        climate = data["climate"]

        row = {
            "District_Name": str(soil["District_Name"]),
            "Soil_color": str(soil["Soil_color"]),
            "Nitrogen": float(soil["Nitrogen"]),
            "Phosphorus": float(soil["Phosphorus"]),
            "Potassium": float(soil["Potassium"]),
            "pH": float(soil["pH"]),
            "Rainfall": float(climate["Rainfall"]),
            "Temperature": float(climate["Temperature"])
        }

        if bundle is None:
            return jsonify({
                "success": False,
                "error": "Model not loaded. Please retrain."
            }), 503

        result = predict(bundle, row)

        # Save user prediction
        try:
            save_user_prediction({
                **row,
                "Crop": result["crop"]["label"],
                "Fertilizer": result["fertilizer"]["label"]
            })
        except Exception as e:
            print("⚠️ Could not save user prediction:", e)

        return jsonify({
            "success": True,
            "prediction": result,
            "model_stats": bundle.get("stats", {}),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/retrain", methods=["POST"])
def retrain_model():
    global bundle, merged_df

    try:
        print("🔄 Retraining model...")

        main_df = load_main_dataset()
        merged_df = main_df.copy()

        if len(merged_df) < 10:
            return jsonify({
                "success": False,
                "error": "Not enough data to retrain"
            }), 400

        bundle = train_models(merged_df)

        return jsonify({
            "success": True,
            "message": f"Model retrained with {len(merged_df)} samples",
            "stats": bundle.get("stats", {})
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/stats", methods=["GET"])
def stats():
    try:
        return jsonify({
            "success": True,
            "stats": get_model_statistics(),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/dataset/info", methods=["GET"])
def dataset_info():
    try:
        if merged_df is None:
            return jsonify({
                "success": False,
                "error": "Dataset not loaded"
            }), 400

        return jsonify({
            "success": True,
            "total_samples": len(merged_df),
            "unique_crops": merged_df["Crop"].nunique(),
            "unique_fertilizers": merged_df["Fertilizer"].nunique(),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "success": True,
        "status": "operational",
        "model_loaded": bundle is not None,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "health": {
            "model_loaded": bundle is not None,
            "dataset_loaded": merged_df is not None and len(merged_df) > 0,
            "timestamp": datetime.now().isoformat()
        }
    })


# -------------------- MAIN --------------------

# if __name__ == "__main__":
#     print("🚀 ML API running on port 5001")
#     print("📌 Endpoints:")
#     print("   POST /predict")
#     print("   POST /retrain")
#     print("   GET  /stats")
#     print("   GET  /dataset/info")
#     print("   GET  /status")
#     print("   GET  /health")

#     app.run(host="0.0.0.0", port=5001, debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
