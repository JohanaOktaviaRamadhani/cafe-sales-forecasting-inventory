import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# STEP 1 — Load Raw Data 
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {os.path.abspath(path)}")
    return pd.read_csv(path)

# STEP 2 — Data Cleaning
def clean_data(data):
    data = data.copy() 
    # Hindari inplace=True untuk kompatibilitas pandas masa depan
    data = data.replace(["ERROR", "UNKNOWN", ""], np.nan)
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["Transaction Date"] = pd.to_datetime(data["Transaction Date"], errors="coerce")
    data = data.dropna(subset=["Transaction Date", "Item", "Quantity"])
    data = data[data["Quantity"] > 0]
    return data[["Transaction Date", "Item", "Quantity"]]

# STEP 3 — Aggregate & Make Continuous (Penting untuk Time Series)
def prepare_time_series(data):
    daily = data.groupby(["Transaction Date", "Item"], as_index=False).agg({"Quantity": "sum"})
    all_items = daily["Item"].unique()
    full_data = []
    
    for item in all_items:
        item_df = daily[daily["Item"] == item].copy()
        dr = pd.date_range(start=item_df["Transaction Date"].min(), 
                           end=item_df["Transaction Date"].max(), freq="D")
        item_df = item_df.set_index("Transaction Date").reindex(dr)
        item_df["Quantity"] = item_df["Quantity"].fillna(0)
        item_df["Item"] = item
        item_df = item_df.reset_index().rename(columns={"index": "Transaction Date"})
        full_data.append(item_df)
    return pd.concat(full_data, ignore_index=True).sort_values(["Item", "Transaction Date"])

# STEP 4 — Feature Engineering
def create_improved_features(data):
    data = data.copy()
    data["Day_of_Week"] = data["Transaction Date"].dt.dayofweek
    data["Is_Weekend"] = data["Transaction Date"].dt.dayofweek.isin([5, 6]).astype(int)
    data["Month"] = data["Transaction Date"].dt.month
    data["Is_Month_End"] = data["Transaction Date"].dt.is_month_end.astype(int)
    
    # Lag & Rolling Features
    data["Lag_1"] = data.groupby("Item")["Quantity"].shift(1)
    data["Lag_2"] = data.groupby("Item")["Quantity"].shift(2)
    data["Lag_3"] = data.groupby("Item")["Quantity"].shift(3)
    data["Lag_7"] = data.groupby("Item")["Quantity"].shift(7)
    
    rolling = data.groupby("Item")["Quantity"].shift(1).rolling(window=7)
    data["Rolling_Mean_7"] = rolling.mean()
    data["Rolling_Std_7"] = rolling.std() 
    
    return data.dropna()

# STEP 5 — Train Model
def train_model_with_cv(df):
    le_item = LabelEncoder()
    df["Item_Encoded"] = le_item.fit_transform(df["Item"])
    
    y = df["Quantity"] 
    feature_cols = [
        "Item_Encoded", "Day_of_Week", "Is_Weekend", "Month", "Is_Month_End",
        "Lag_1", "Lag_2", "Lag_3", "Lag_7", "Rolling_Mean_7", "Rolling_Std_7"
    ]
    X = df[feature_cols]
    num_cols = ["Lag_1", "Lag_2", "Lag_3", "Lag_7", "Rolling_Mean_7", "Rolling_Std_7"]
    
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    print("Memulai Cross-Validation...")
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
        
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        preds = model.predict(X_test_scaled)
        cv_scores.append(mean_absolute_error(y_test, preds))

    print(f"Rata-rata MAE (CV): {np.mean(cv_scores):.4f}")

    # FINAL TRAINING
    scaler_final = StandardScaler()
    X_scaled = X.copy()
    X_scaled[num_cols] = scaler_final.fit_transform(X[num_cols])
    
    model_final = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model_final.fit(X_scaled, y)
    
    # Path handling: simpan ke folder models di root
    model_path = "models"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    joblib.dump(model_final, f"{model_path}/rf_model.pkl")
    joblib.dump(le_item, f"{model_path}/item_encoder.pkl")
    joblib.dump(scaler_final, f"{model_path}/scaler.pkl")
    print(f"✨ Model & Artifacts sukses disimpan di folder /{model_path}!")

if __name__ == "__main__":
    DATA_PATH = "data/dirty_cafe_sales.csv"
    if os.path.exists(DATA_PATH):
        df_raw = load_data(DATA_PATH)
        df_clean = clean_data(df_raw)
        df_ts = prepare_time_series(df_clean)
        df_feat = create_improved_features(df_ts)
        train_model_with_cv(df_feat)
    else:
        print(f"❌ Error: File {DATA_PATH} tidak ditemukan.")