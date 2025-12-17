# clean_dataset.py
# Run this to automatically clean your datasets

import pandas as pd
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
USER_DATASET_PATH = os.path.join(BASE_DIR, "data", "predicted_data.csv")

def clean_dataset(filepath, backup=True):
    """Clean a dataset by removing NaN values"""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    print(f"\n🔧 Cleaning: {filepath}")
    
    # Load dataset
    df = pd.read_csv(filepath)
    original_size = len(df)
    print(f"📊 Original size: {original_size} rows")
    
    # Count NaN values
    nan_count = df.isna().sum().sum()
    if nan_count == 0:
        print("✅ No NaN values found - dataset is clean!")
        return True
    
    print(f"⚠️  Found {nan_count} NaN values")
    
    # Show which columns have NaN
    nan_cols = df.isna().sum()
    for col, count in nan_cols[nan_cols > 0].items():
        print(f"   - {col}: {count} NaN values")
    
    # Create backup if requested
    if backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.replace(".csv", f"_backup_{timestamp}.csv")
        df.to_csv(backup_path, index=False)
        print(f"💾 Backup created: {backup_path}")
    
    # Remove rows with NaN
    df_clean = df.dropna()
    cleaned_size = len(df_clean)
    removed = original_size - cleaned_size
    
    print(f"🧹 Removed {removed} rows with NaN values")
    print(f"✅ Clean dataset: {cleaned_size} rows")
    
    # Save cleaned dataset
    df_clean.to_csv(filepath, index=False)
    print(f"💾 Saved cleaned dataset to: {filepath}")
    
    return True

print("="*60)
print("🧹 DATASET CLEANUP TOOL")
print("="*60)

# Clean main dataset
print("\n1️⃣  Cleaning Main Dataset...")
clean_dataset(DATASET_PATH, backup=True)

# Clean user dataset
print("\n2️⃣  Cleaning User Dataset...")
if os.path.exists(USER_DATASET_PATH) and os.path.getsize(USER_DATASET_PATH) > 0:
    clean_dataset(USER_DATASET_PATH, backup=True)
else:
    print("ℹ️  User dataset is empty - skipping")

print("\n" + "="*60)
print("✅ CLEANUP COMPLETE!")
print("="*60)
print("\n💡 Next Steps:")
print("   1. Verify your datasets are clean")
print("   2. Run: python run_retrain.py")
print("   3. Restart ml_service.py")
print("="*60)