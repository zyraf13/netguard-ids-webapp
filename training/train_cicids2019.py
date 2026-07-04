"""
train_cicids2019.py

Cara pakai:
1. Taruh file syn_data.csv di folder ../data/cicids2019/
   (download manual dari Kaggle: tarundhamor/cicids-2019-dataset,
   karena environment training tidak selalu punya akses otomatis ke Kaggle)
2. Jalankan: python train_cicids2019.py
3. Hasil tersimpan di ../artifacts/cicids2019/
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from common import train_and_select_best, save_artifacts

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cicids2019")
DATASET_ID = "cicids2019"


def load_data():
    data_path = os.path.join(DATA_DIR, "syn_data.csv")
    df = pd.read_csv(data_path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def main():
    df = load_data()
    df.dropna(inplace=True)

    # Label: BENIGN -> 0, selain itu -> 1 (serangan)
    df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

    X = df.drop(columns=["Label"])
    y = df["Label"]

    # Dataset ini punya banyak kolom non-numerik/campuran; ambil numerik saja
    X = X.select_dtypes(include=[np.number])
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    y = y.loc[X.index]

    feature_names = X.columns
    encoders = {}  # tidak ada kolom kategorikal yang dipakai untuk dataset ini

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    best_name, best_model, results = train_and_select_best(
        X_scaled, y, DATASET_ID
    )

    save_artifacts(
        dataset_id=DATASET_ID,
        best_name=best_name,
        best_model=best_model,
        scaler=scaler,
        encoders=encoders,
        feature_names=feature_names,
        label_meaning={"0": "Benign", "1": "Attack"},
        all_results=results,
    )


if __name__ == "__main__":
    main()
