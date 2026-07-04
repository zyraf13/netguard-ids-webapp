"""
common.py
Fungsi bantu bersama untuk semua script training (UNSW-NB15, NSL-KDD, CICIDS2019).
Tugasnya: melatih beberapa model, memilih model terbaik (berdasarkan F1-score),
lalu menyimpan model + scaler + encoder + metadata fitur ke folder artifacts/.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def get_models():
    """Model yang sama seperti di notebook penelitian."""
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }


def train_and_select_best(X, y, dataset_name, test_size=0.2):
    """
    Melatih semua model pada satu split train/test, mengembalikan:
    - nama model terbaik
    - objek model terbaik
    - dict metrik semua model (untuk ditampilkan di dashboard web nanti)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    results = {}
    best_name, best_model, best_f1 = None, None, -1

    for name, model in get_models().items():
        print(f"[{dataset_name}] Training {name} ...")
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.time() - start

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "training_time_sec": elapsed,
            "confusion_matrix": cm,
        }
        print(
            f"    acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} "
            f"f1={f1:.4f} time={elapsed:.2f}s"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    return best_name, best_model, results


def save_artifacts(dataset_id, best_name, best_model, scaler, encoders, feature_names,
                    label_meaning, all_results):
    """
    Menyimpan semua yang dibutuhkan backend untuk melakukan prediksi nanti:
    - model terbaik (.joblib)
    - scaler (.joblib)
    - encoder kategorikal per kolom (.joblib, dict)
    - metadata.json (nama fitur, model terbaik, arti label, hasil evaluasi semua model)
    """
    out_dir = os.path.join(ARTIFACT_DIR, dataset_id)
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(out_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    joblib.dump(encoders, os.path.join(out_dir, "encoders.joblib"))

    metadata = {
        "dataset_id": dataset_id,
        "best_model": best_name,
        "feature_names": list(feature_names),
        "categorical_features": list(encoders.keys()),
        "label_meaning": label_meaning,  # contoh: {"0": "Normal", "1": "Attack"}
        "evaluation": all_results,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[{dataset_id}] Artifacts tersimpan di: {out_dir}")
    print(f"[{dataset_id}] Model terbaik: {best_name}")
