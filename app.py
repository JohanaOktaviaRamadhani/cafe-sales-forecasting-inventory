import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- CONFIG DASHBOARD ---
st.set_page_config(page_title="Cafe Sales Intelligence", layout="wide")

# --- 1. LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/rf_model.pkl")
    le_item = joblib.load("models/item_encoder.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, le_item, scaler

# --- 2. LOAD & PREPROCESS DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/dirty_cafe_sales.csv")
    df.replace(["ERROR", "UNKNOWN", ""], np.nan, inplace=True)    
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df.dropna(subset=["Transaction Date", "Item", "Quantity"], inplace=True)
    
    if 'Price Per Unit' in df.columns:
        df['Price Per Unit'] = pd.to_numeric(df['Price Per Unit'], errors='coerce').fillna(3.0)
    else:
        prices = {'Coffee': 2.0, 'Cake': 3.0, 'Cookie': 1.0, 'Salad': 5.0, 'Smoothie': 4.0}
        df['Price Per Unit'] = df['Item'].map(prices).fillna(3.0)
    
    df["Total Sales"] = df["Quantity"] * df["Price Per Unit"]
    return df

# --- 3. LOGIKA FORECAST ---
def run_forecast(item_name, df_raw, model, le_item, scaler):
    # Agregasi harian
    df_item = df_raw[df_raw["Item"] == item_name].groupby("Transaction Date")["Quantity"].sum().reset_index()
    df_item = df_item.sort_values("Transaction Date")

    # Kita butuh minimal 7-14 hari terakhir untuk hitung Lag & Rolling
    df_hist = df_item.tail(14).copy()
    last_date = df_hist["Transaction Date"].max()
    
    # Penampung data untuk update rekursif
    history_values = df_hist["Quantity"].tolist()
    forecasts = []
    
    # Loop 7 hari ke depan
    for i in range(1, 8):
        f_date = last_date + pd.Timedelta(days=i)
        
        # A. Fitur Temporal Baru
        dow = f_date.dayofweek
        is_weekend = 1 if dow in [5, 6] else 0
        month = f_date.month
        is_month_end = 1 if f_date.is_month_end else 0
        
        # B. Fitur Lag (Sesuai train_model.py)
        lag1 = history_values[-1]
        lag2 = history_values[-2] if len(history_values) >= 2 else lag1
        lag3 = history_values[-3] if len(history_values) >= 3 else lag2
        lag7 = history_values[-7] if len(history_values) >= 7 else history_values[0]
        
        # C. Rolling Stats
        last_7_days = history_values[-7:]
        roll_mean = np.mean(last_7_days)
        roll_std = np.std(last_7_days) if len(last_7_days) > 1 else 0
        
        # D. Gabungkan ke DataFrame Input
        inp = pd.DataFrame([{
            "Item_Encoded": le_item.transform([item_name])[0],
            "Day_of_Week": dow,
            "Is_Weekend": is_weekend,
            "Month": month,
            "Is_Month_End": is_month_end,
            "Lag_1": lag1, "Lag_2": lag2, "Lag_3": lag3, "Lag_7": lag7,
            "Rolling_Mean_7": roll_mean, "Rolling_Std_7": roll_std
        }])
        
        # E. Scaling (Hanya kolom numerik sesuai scaler)
        num_cols = ["Lag_1", "Lag_2", "Lag_3", "Lag_7", "Rolling_Mean_7", "Rolling_Std_7"]
        inp[num_cols] = scaler.transform(inp[num_cols])
        
        # F. Predict & Inverse Log (np.expm1)
        pred_val = model.predict(inp)[0]
        pred_val = max(0, pred_val)
        
        forecasts.append({"Transaction Date": f_date, "Quantity": pred_val})
        history_values.append(pred_val)
    return df_hist.tail(7), pd.DataFrame(forecasts)

# --- EXECUTION ---
model, le_item, scaler = load_artifacts()
df_raw = load_data()
max_date = df_raw["Transaction Date"].max()
all_items = sorted(df_raw["Item"].unique())

MAE_SCORE = 4.5

# Pre-calculate untuk efisiensi
results = {}
for it in all_items:
    h, f = run_forecast(it, df_raw, model, le_item, scaler)
    change = ((f["Quantity"].mean() - h["Quantity"].mean()) / h["Quantity"].mean() * 100) if h["Quantity"].mean() > 0 else 0
    results[it] = (h, f, change)

# --- HEADER & KPI ---
st.title("☕ Cafe Sales Strategic Dashboard")
st.markdown(f"**Analisis Periode: {max_date.strftime('%B %Y')}**")

# Metrik Utama
# 1. Data Bulan Sekarang
df_month = df_raw[df_raw["Transaction Date"].dt.month == max_date.month]

# 2. Data Bulan Sebelumnya (untuk komparasi)
prev_month_date = max_date - pd.DateOffset(months=1)
df_prev_month = df_raw[df_raw["Transaction Date"].dt.month == prev_month_date.month]

# KPI 1: Most Popular & Perbandingannya
pop_item = df_month.groupby("Item")["Quantity"].sum().idxmax()
pop_val = df_month.groupby("Item")["Quantity"].sum().max()

# Hitung berapa Juice terjual bulan lalu
pop_val_prev = df_prev_month[df_prev_month["Item"] == pop_item]["Quantity"].sum() if not df_prev_month.empty else 0
diff_pop = pop_val - pop_val_prev

# KPI 2: Highest Revenue & Perbandingannya
inc_item = df_month.groupby("Item")["Total Sales"].sum().idxmax()
inc_val = df_month.groupby("Item")["Total Sales"].sum().max()
# Hitung berapa income Smoothie bulan lalu
inc_val_prev = df_prev_month[df_prev_month["Item"] == inc_item]["Total Sales"].sum() if not df_prev_month.empty else 0
diff_inc = inc_val - inc_val_prev

# KPI 3: Next Week Star (Sudah ada di loop results)
star_item = max(results, key=lambda k: results[k][2])
star_growth = results[star_item][2]

k1, k2, k3 = st.columns(3)
k1.metric(
    label=f"🏆 MOST POPULAR: {pop_item}", 
    value=f"{int(pop_val)} Units", 
    delta=f"{int(diff_pop)} dibandingkan Bulan Lalu",
    delta_color="normal" 
)
k2.metric(
    label=f"💰 TOP REVENUE: {inc_item}", 
    value=f"${inc_val:,.2f}", 
    delta=f"${diff_inc:,.2f} dibandingkan Bulan Lalu",
    delta_color="normal"
)
k3.metric(
    label=f"🌟 NEXT WEEK STAR: {star_item}", 
    value=f"{star_growth:+.1f}%", 
    delta="Prediksi kenaikan minggu depan",
    delta_color="normal"
)
st.divider()

# --- GRID PERFORMANCE ---
rows = [all_items[i:i + 3] for i in range(0, len(all_items), 3)]
for row in rows:
    cols = st.columns(3)
    for i, item_name in enumerate(row):
        with cols[i]:
            with st.container(border=True):
                st.subheader(item_name)
                df_h, df_f, perc = results[item_name]
                
                # Plotting
                fig, ax = plt.subplots(figsize=(6, 4))
                xh = df_h["Transaction Date"].dt.strftime('%d/%m')
                xf = df_f["Transaction Date"].dt.strftime('%d/%m')
                ax.plot(xh, df_h["Quantity"], marker='o', color='#1f77b4', label='Histori')
                ax.plot(xf, df_f["Quantity"], marker='s', linestyle='--', color='#ff7f0e', label='Prediksi')                
                for x, y in zip(xh, df_h["Quantity"]): ax.text(x, y + 0.3, f'{int(y)}', ha='center', fontsize=8, color='blue')
                for x, y in zip(xf, df_f["Quantity"]): ax.text(x, y + 0.3, f'{int(y)}', ha='center', fontsize=8, color='orange')
                plt.xticks(rotation=45)
                ax.legend(prop={'size': 8})
                st.pyplot(fig)
                plt.close(fig)
                
                # Insight
                st.write("---")
                if perc > 0:
                    st.success(f"**Prediksi: Bakal Laris (+{perc:.1f}%)**")
                    st.markdown(f"**Karena** Ada pola kenaikan yang konsisten. <br>**Saran:** Siapkan stok lebih awal supaya tidak keteteran saat jam ramai.", unsafe_allow_html=True)
                elif perc < 0:
                    st.warning(f"**Prediksi: Bakal Agak Sepi ({perc:.1f}%)**")
                    st.markdown(f"**Karena** Permintaan lagi melandai. <br>**Saran:** Coba buat promo bundling atau diskon kecil untuk menarik minat.", unsafe_allow_html=True)
                else:
                    st.info("**Prediksi: Stabil**")
                    st.markdown(f"**Karena** Penjualan terjaga konsisten. <br>**Saran:** Pertahankan kualitas dan ketersediaan stok seperti biasa.", unsafe_allow_html=True)
# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 📊 Model Analytics") 
    st.divider()

    # 1. Model Specification
    st.markdown("### 🎯 Model Specification")
    st.markdown(f"""
    * Algorithm: {model.__class__.__name__}
    * CV MAE: {MAE_SCORE} units
    * Estimators: {model.n_estimators}
    * Max Depth: {model.max_depth}
    """)
    st.divider()

    # 2. Training Setup
    st.markdown("### Training Setup")
    st.markdown(f"""
    * Cross Validation: TimeSeriesSplit
    * Folds: 5-Fold
    * Training Set: Sequential Split (Chronological)
    * Features: Temporal, Lag (1,2,3,7), Rolling Stats
    """)
    st.divider()

    # 3. Business Context
    st.markdown("### Business Context")
    st.markdown("**Tujuan:**")
    st.write("Mengoptimalkan manajemen stok harian dan meminimalkan kerugian akibat bahan baku kadaluarsa (waste).")
    st.markdown("**Kegunaan:**")
    st.write("Membantu tim operasional menentukan kuantitas produksi harian berdasarkan pola historis dan tren mingguan.")
    st.divider()

    # 4. Feature Importance
    st.markdown("### Feature Importance")
    feature_cols = [
        "Item_Encoded", "Day_of_Week", "Is_Weekend", "Month", "Is_Month_End",
        "Lag_1", "Lag_2", "Lag_3", "Lag_7", "Rolling_Mean_7", "Rolling_Std_7"
    ]
    df_importance = pd.DataFrame({
        'Feature': [f.replace('_', ' ') for f in feature_cols],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    st.bar_chart(df_importance.set_index('Feature'), horizontal=True)
    st.caption("Faktor dominan menunjukkan pengaruh besar data historis terhadap prediksi.")
    