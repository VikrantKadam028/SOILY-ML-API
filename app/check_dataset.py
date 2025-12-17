# check_dataset.py
# Save this file in the app/ directory and run it

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
USER_DATASET_PATH = os.path.join(BASE_DIR, "data", "predicted_data.csv")

print("="*60)
print("🔍 DATASET DIAGNOSTIC TOOL")
print("="*60)

# Check main dataset
print("\n📊 Main Dataset Analysis:")
print(f"Path: {DATASET_PATH}")

if os.path.exists(DATASET_PATH):
    df_main = pd.read_csv(DATASET_PATH)
    print(f"✅ File exists: {len(df_main)} rows")
    
    print(f"\n📋 Columns: {list(df_main.columns)}")
    
    # Check for NaN values
    print(f"\n🔍 NaN Values:")
    nan_counts = df_main.isna().sum()
    total_nan = nan_counts.sum()
    
    if total_nan > 0:
        print(f"⚠️  Found {total_nan} total NaN values:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"   - {col}: {count} NaN values ({count/len(df_main)*100:.1f}%)")
    else:
        print("✅ No NaN values found")
    
    # Check data types
    print(f"\n📊 Data Types:")
    for col, dtype in df_main.dtypes.items():
        print(f"   - {col}: {dtype}")
    
    # Check numeric ranges
    print(f"\n📈 Numeric Ranges:")
    for col in ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']:
        if col in df_main.columns:
            print(f"   - {col}: {df_main[col].min():.2f} to {df_main[col].max():.2f} (mean: {df_main[col].mean():.2f})")
    
    # Check categorical values
    print(f"\n🏷️  Categorical Values:")
    print(f"   - Unique Crops: {df_main['Crop'].nunique()}")
    print(f"   - Unique Fertilizers: {df_main['Fertilizer'].nunique()}")
    print(f"   - Unique Districts: {df_main['District_Name'].nunique()}")
    print(f"   - Unique Soil Colors: {df_main['Soil_color'].nunique()}")
    
    # Show top crops
    print(f"\n🌾 Top 10 Crops:")
    for crop, count in df_main['Crop'].value_counts().head(10).items():
        print(f"   - {crop}: {count}")
    
    # Show top fertilizers
    print(f"\n💊 Top 10 Fertilizers:")
    for fert, count in df_main['Fertilizer'].value_counts().head(10).items():
        print(f"   - {fert}: {count}")
    
    # Check for duplicates
    duplicates = df_main.duplicated().sum()
    print(f"\n🔄 Duplicate rows: {duplicates}")
    
else:
    print("❌ Main dataset file not found!")

# Check user dataset
print("\n" + "="*60)
print("📊 User Dataset Analysis:")
print(f"Path: {USER_DATASET_PATH}")

if os.path.exists(USER_DATASET_PATH) and os.path.getsize(USER_DATASET_PATH) > 0:
    df_user = pd.read_csv(USER_DATASET_PATH)
    print(f"✅ File exists: {len(df_user)} rows")
    
    # Check for NaN in user data
    user_nan = df_user.isna().sum().sum()
    if user_nan > 0:
        print(f"⚠️  User dataset has {user_nan} NaN values!")
        print(df_user.isna().sum())
    else:
        print("✅ No NaN values in user dataset")
else:
    print("ℹ️  User dataset is empty or doesn't exist")

# Recommendations
print("\n" + "="*60)
print("💡 RECOMMENDATIONS:")
print("="*60)

if total_nan > 0:
    print("⚠️  Action Required:")
    print("   1. Your dataset has NaN values that need to be cleaned")
    print("   2. Option A: Clean your CSV file manually (remove rows with empty cells)")
    print("   3. Option B: Update model_utils.py to handle NaN automatically (use the fix provided)")
    print("\n   Quick Fix Command:")
    print(f"   python -c \"import pandas as pd; df = pd.read_csv('{DATASET_PATH}'); df.dropna().to_csv('{DATASET_PATH}', index=False); print('Cleaned!')\"")
else:
    print("✅ Dataset looks good! You can proceed with retraining.")

print("\n" + "="*60)