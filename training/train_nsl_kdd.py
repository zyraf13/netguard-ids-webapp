"""
train_nsl_kdd.py

Cara pakai:
1. Taruh file berikut di folder ../data/nsl_kdd/:
   - Train_data.csv
   - Test_data.csv
2. Jalankan: python training/train_nsl_kdd.py
3. Hasil tersimpan di ../artifacts/nsl_kdd/
"""

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from common import train_and_select_best, save_artifacts

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "nsl_kdd")
DATASET_ID = "nsl_kdd"


def read_csv_flexible(path):
    """
    Membaca CSV dengan beberapa kemungkinan delimiter.
    Cocok untuk file NSL-KDD yang kadang pakai ; atau ,.
    """
    for sep in [";", ","]:
        try:
            df = pd.read_csv(path, sep=sep)
            df.columns = df.columns.str.strip()

            # Kalau cuma 1 kolom, kemungkinan delimiter salah
            if len(df.columns) > 1:
                print(f"Berhasil membaca {os.path.basename(path)} dengan delimiter '{sep}'")
                print("Contoh kolom:", df.columns.tolist()[:10])
                return df
        except Exception as e:
            print(f"Gagal membaca {os.path.basename(path)} dengan delimiter '{sep}': {e}")

    # Percobaan terakhir: auto-detect delimiter
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    print(f"Berhasil membaca {os.path.basename(path)} dengan auto-detect delimiter")
    print("Contoh kolom:", df.columns.tolist()[:10])
    return df


def load_data():
    train_path = os.path.join(DATA_DIR, "Train_data.csv")
    test_path = os.path.join(DATA_DIR, "Test_data.csv")

    train_df = read_csv_flexible(train_path)
    test_df = read_csv_flexible(test_path)

    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # Bersihkan nama kolom dari spasi
    df.columns = df.columns.str.strip()

    print("Jumlah kolom:", len(df.columns))
    print("Daftar kolom:", df.columns.tolist())

    return df


def main():
    df = load_data()
    df.dropna(inplace=True)

    required_cols = ["protocol_type", "service", "flag", "class"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Kolom berikut tidak ditemukan: {missing_cols}\n"
            f"Kolom yang terbaca dari file adalah: {df.columns.tolist()}"
        )

    categorical_cols = ["protocol_type", "service", "flag", "class"]
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Kolom numerik yang mungkin memakai koma sebagai desimal
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)

    X = df.drop(columns=["class"])
    y = df["class"]
    feature_names = X.columns

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    best_name, best_model, results = train_and_select_best(
        X_scaled, y, DATASET_ID
    )

    class_encoder = encoders["class"]
    label_meaning = {
        str(i): str(label) for i, label in enumerate(class_encoder.classes_)
    }

    save_artifacts(
        dataset_id=DATASET_ID,
        best_name=best_name,
        best_model=best_model,
        scaler=scaler,
        encoders=encoders,
        feature_names=feature_names,
        label_meaning=label_meaning,
        all_results=results,
    )


if __name__ == "__main__":
    main()