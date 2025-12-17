# wsgi.py
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml_service import app as application

# Load environment variables if using python-dotenv
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print("✅ Environment variables loaded")
except ImportError:
    pass
except Exception as e:
    print(f"⚠️  Could not load environment variables: {e}")

print("🚀 WSGI application initialized")