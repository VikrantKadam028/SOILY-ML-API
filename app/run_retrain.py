# run_retrain.py
import os
import requests

ML_HOST = os.environ.get("ML_HOST", "http://127.0.0.1:5001")
RETRAIN_URL = f"{ML_HOST}/retrain"

if __name__ == "__main__":
    try:
        r = requests.post(RETRAIN_URL, timeout=300)
        print("Status code:", r.status_code)
        print("Response:", r.json())
    except Exception as e:
        print("Retrain call failed:", e)
