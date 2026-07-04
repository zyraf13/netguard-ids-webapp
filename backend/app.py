"""
app.py — Backend API untuk web deteksi intrusi jaringan.

Menjalankan:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000

Endpoint:
    GET  /api/datasets                     -> daftar dataset yang tersedia (yang sudah ada artifact-nya)
    GET  /api/datasets/{dataset_id}         -> detail: fitur, model terbaik, evaluasi semua model
    POST /api/datasets/{dataset_id}/predict -> prediksi 1 baris data baru
"""

import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

app = FastAPI(title="Network Intrusion Detection API")

# Izinkan frontend (file HTML statis / origin lain) memanggil API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache artifact yang sudah di-load supaya tidak baca file tiap request
_cache: Dict[str, Dict[str, Any]] = {}


def load_dataset_artifacts(dataset_id: str) -> Dict[str, Any]:
    if dataset_id in _cache:
        return _cache[dataset_id]

    ds_dir = os.path.join(ARTIFACT_DIR, dataset_id)
    metadata_path = os.path.join(ds_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' belum tersedia. Jalankan script training-nya dulu.",
        )

    with open(metadata_path) as f:
        metadata = json.load(f)

    model = joblib.load(os.path.join(ds_dir, "model.joblib"))
    scaler = joblib.load(os.path.join(ds_dir, "scaler.joblib"))
    encoders = joblib.load(os.path.join(ds_dir, "encoders.joblib"))

    bundle = {
        "metadata": metadata,
        "model": model,
        "scaler": scaler,
        "encoders": encoders,
    }
    _cache[dataset_id] = bundle
    return bundle


class PredictRequest(BaseModel):
    # Contoh: {"proto": "tcp", "service": "http", "dur": 0.5, ...}
    features: Dict[str, Any]


@app.get("/api/datasets")
def list_datasets():
    if not os.path.isdir(ARTIFACT_DIR):
        return {"datasets": []}
    available = []
    for name in sorted(os.listdir(ARTIFACT_DIR)):
        meta_path = os.path.join(ARTIFACT_DIR, name, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            available.append(
                {"dataset_id": name, "best_model": meta["best_model"]}
            )
    return {"datasets": available}


@app.get("/api/datasets/{dataset_id}")
def get_dataset_detail(dataset_id: str):
    bundle = load_dataset_artifacts(dataset_id)
    return bundle["metadata"]


@app.post("/api/datasets/{dataset_id}/predict")
def predict(dataset_id: str, req: PredictRequest):
    bundle = load_dataset_artifacts(dataset_id)
    metadata = bundle["metadata"]
    model = bundle["model"]
    scaler = bundle["scaler"]
    encoders = bundle["encoders"]

    feature_names = metadata["feature_names"]
    missing = [f for f in feature_names if f not in req.features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Fitur berikut belum diisi: {missing}",
        )

    row = {}
    for f in feature_names:
        value = req.features[f]
        if f in encoders:
            le = encoders[f]
            try:
                value = le.transform([value])[0]
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Nilai '{value}' untuk fitur '{f}' tidak dikenali. "
                    f"Pilihan valid: {list(le.classes_)}",
                )
        row[f] = value

    X = pd.DataFrame([row], columns=feature_names)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    pred_label = metadata["label_meaning"].get(str(int(pred)), str(pred))

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0].tolist()

    return {
        "dataset_id": dataset_id,
        "model_used": metadata["best_model"],
        "prediction_raw": int(pred),
        "prediction_label": pred_label,
        "probabilities": proba,
    }
