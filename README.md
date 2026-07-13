# NetGuard IDS Web App

NetGuard merupakan aplikasi web berbasis **Machine Learning** yang dirancang untuk mendeteksi intrusi, serangan siber, dan anomali pada lalu lintas jaringan.

Aplikasi ini dikembangkan dari penelitian *Machine Learning-Based Network Intrusion Detection System* dengan membandingkan tiga algoritma klasifikasi, yaitu **K-Nearest Neighbors (KNN)**, **Random Forest**, dan **XGBoost** pada dataset **UNSW-NB15**, **NSL-KDD**, dan **CICIDS 2019**.

Setiap model dievaluasi menggunakan metrik **accuracy, precision, recall, dan F1-score**. Model dengan F1-score terbaik pada masing-masing dataset dipilih sebagai model utama untuk melakukan prediksi melalui aplikasi web.

NetGuard terdiri dari backend API berbasis **FastAPI** dan frontend dashboard yang memungkinkan pengguna memilih dataset, melihat perbandingan performa model, memasukkan karakteristik lalu lintas jaringan, dan memperoleh hasil prediksi berupa **Normal** atau **Attack**.  

## Fitur Utama

- Mendukung dataset UNSW-NB15, NSL-KDD, dan CICIDS 2019.
- Pra-pemrosesan data numerik dan kategorikal.
- Implementasi algoritma KNN, Random Forest, dan XGBoost.
- Pemilihan model terbaik berdasarkan F1-score.
- Perbandingan accuracy, precision, recall, dan F1-score.
- Form prediksi yang menyesuaikan fitur setiap dataset.
- Prediksi lalu lintas jaringan sebagai Normal atau Attack.
- Penyimpanan model, scaler, encoder, dan metadata menggunakan Joblib.
- Backend REST API menggunakan FastAPI.
- Dashboard web untuk visualisasi hasil dan proses inferensi.

## Teknologi yang Digunakan

### Machine Learning
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib

### Backend
- FastAPI
- Uvicorn
- Pydantic

### Frontend
- HTML
- CSS
- JavaScript

Daftar dependency backend pada repository mencakup FastAPI, Uvicorn, Pandas, NumPy, Scikit-learn, XGBoost, Joblib, dan Pydantic. 

## Alur Sistem

1. Dataset dimuat dan dibersihkan.
2. Fitur kategorikal diubah menjadi data numerik.
3. Fitur numerik dinormalisasi.
4. Data digunakan untuk melatih KNN, Random Forest, dan XGBoost.
5. Setiap model dievaluasi menggunakan beberapa metrik klasifikasi.
6. Model dengan F1-score tertinggi disimpan sebagai model utama.
7. Pengguna memasukkan data lalu lintas jaringan melalui dashboard.
8. Backend memproses input dan menghasilkan prediksi Normal atau Attack.

## Tujuan Pengembangan

Proyek ini dikembangkan sebagai implementasi hasil penelitian Machine Learning ke dalam aplikasi web yang lebih interaktif dan mudah digunakan. Aplikasi ini juga menjadi dasar untuk pengembangan sistem deteksi intrusi jaringan yang lebih lanjut, seperti monitoring lalu lintas secara real-time, klasifikasi jenis serangan, integrasi log jaringan, dan dashboard keamanan siber.
