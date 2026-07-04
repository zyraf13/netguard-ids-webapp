"""
train_unsw_nb15.py

Cara pakai:
1. Taruh file berikut di folder ../data/unsw_nb15/:
   - UNSW_NB15_training-set.csv
   - UNSW_NB15_testing-set.csv
2. Jalankan: python train_unsw_nb15.py
3. Hasil (model, scaler, encoder, metadata) tersimpan di ../artifacts/unsw_nb15/
"""

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from common import train_and_select_best, save_artifacts

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "unsw_nb15")
DATASET_ID = "unsw_nb15"


def load_data():
    train_path = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    df = pd.concat([train_df, test_df], axis=0)
    df.drop(columns=["id"], inplace=True, errors="ignore")
    return df


def main():
    df = load_data()
    df.dropna(inplace=True)

    categorical_cols = ["proto", "service", "state"]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=["label", "attack_cat"])
    y = df["label"]
    feature_names = X.columns

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
        label_meaning={"0": "Normal", "1": "Attack"},
        all_results=results,
    )


if __name__ == "__main__":
    main()
