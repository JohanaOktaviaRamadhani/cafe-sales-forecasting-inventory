# ☕ Cafe Sales Forecasting & Smart Inventory System

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![ML Framework](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Dashboard](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

## Executive Summary
Proyek ini mengintegrasikan **Predictive Analytics** ke dalam manajemen operasional kafe. Fokus utamanya adalah mentransformasi data transaksi historis yang *noisy* menjadi instruksi inventaris yang presisi. Dengan menggunakan pendekatan Machine Learning, sistem ini membantu pemilik bisnis meminimalkan *waste* (pemborosan bahan) dan *stockout* (kehilangan potensi penjualan).

---

## The Business Challenge
Pada bisnis F&B skala menengah-kecil, fluktuasi harian seringkali dianggap acak. Masalah utamanya meliputi:
* **Overstocking:** Bahan baku *perishable* terbuang karena prediksi manual yang terlalu optimis.
* **Lost Sales:** Kehilangan profit saat permintaan melonjak namun stok tidak mencukupi.
* **Intuition Bias:** Keputusan belanja bahan baku masih berbasis insting, bukan data tren musiman.

---

## Data Science Workflow

### 1. Feature Engineering (Signal Extraction)
Model tidak hanya melihat angka penjualan, tetapi mengekstraksi konteks melalui:
* **Temporal Features:** Encoding hari (Day of Week), status akhir pekan (Is_Weekend), dan pola akhir bulan (Is_Month_End).
* **Autoregressive Lags:** Menggunakan `Lag_1`, `Lag_3`, dan `Lag_7` untuk menangkap korelasi siklus mingguan.
* **Statistical Windows:** Mengimplementasikan `Rolling Mean` dan `Rolling Std` (7 hari) untuk menangkap volatilitas dan tren jangka pendek.

### 2. Model Architecture
Sistem ini menggunakan algoritma **Random Forest Regressor**. 
* **Why RF?** Algoritma ini sangat tangguh terhadap pencilan (*outliers*) dan mampu menangkap hubungan non-linear antara fitur temporal dengan volume penjualan.
* **Preprocessing:** Fitur numerik distandarisasi menggunakan `StandardScaler` untuk menjaga stabilitas estimasi bobot fitur.

### 3. Validation Strategy
Menggunakan **Time Series Cross-Validation** (Walk-forward Validation) dengan 5-fold split untuk menghindari *data leakage*.
* **Performance Metric:** Mencapai **Mean Absolute Error (MAE) ~4.5**, yang berarti rata-rata selisih prediksi hanya sekitar 4-5 unit per item.

---

## Smart Inventory Logic
Data Science di sini digunakan untuk mendukung **Prescriptive Analytics**. Saya menerapkan rumus *Safety Stock* ke dalam logika sistem:

$$Stock\ Recommendation = \hat{y} (Forecast) + \sigma (Safety\ Buffer)$$

Sistem secara otomatis menambahkan buffer **+5 unit** sebagai jaring pengaman operasional untuk menghadapi lonjakan pembeli tak terduga.

---

## Dashboard Features
1.  **Strategic KPIs:** Monitoring produk terlaris dan total omzet bulanan.
2.  **Interactive Forecast Charts:** Visualisasi komparatif antara data aktual dan proyeksi AI 7 hari ke depan.
3.  **Operational Instruction:** Mengubah angka prediksi menjadi perintah belanja yang jelas (Contoh: "Sedia 12 Unit").
4.  **Trend Indicators:** Deteksi otomatis apakah tren penjualan sedang naik atau turun secara signifikan.


