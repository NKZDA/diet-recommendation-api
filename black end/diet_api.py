# === diet_api.py ===
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import pandas as pd
import io
from typing import Dict, Any, List
from pathlib import Path

# -------------------------------
# โหลดโมเดล + label encoder (ใช้ path ที่แน่นอน)
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "diet_recommendation_rf_model.joblib")
label_encoder = joblib.load(BASE_DIR / "label_encoder.joblib")

# -------------------------------
# สร้าง FastAPI app
# -------------------------------
app = FastAPI(title="Diet Recommendation API", version="1.0")

# -------------------------------
# Data Models
# -------------------------------
class Record(BaseModel):
    data: Dict[str, Any]

class Records(BaseModel):
    records: List[Dict[str, Any]]

# -------------------------------
# Routes
# -------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "classes": list(label_encoder.classes_),
        "n_classes": len(label_encoder.classes_)
    }

@app.post("/predict-one")
def predict_one(record: Record):
    df = pd.DataFrame([record.data])
    pred_num = model.predict(df)[0]
    pred_lbl = label_encoder.inverse_transform([pred_num])[0]
    proba = model.predict_proba(df)[0]
    return {
        "prediction": pred_lbl,
        "probabilities": {str(c): float(proba[i]) for i, c in enumerate(label_encoder.classes_)}
    }

@app.post("/predict")
def predict_many(records: Records):
    df = pd.DataFrame(records.records)
    pred_num = model.predict(df)
    pred_lbl = label_encoder.inverse_transform(pred_num)
    proba = model.predict_proba(df)
    results = []
    for i in range(len(df)):
        results.append({
            "prediction": pred_lbl[i],
            "probabilities": {str(c): float(proba[i, j]) for j, c in enumerate(label_encoder.classes_)}
        })
    return {"count": len(results), "results": results}

@app.post("/predict-csv")
def predict_csv(file: UploadFile = File(...)):
    content = file.file.read()
    df = pd.read_csv(io.BytesIO(content))
    pred_num = model.predict(df)
    pred_lbl = label_encoder.inverse_transform(pred_num)
    proba = model.predict_proba(df)
    results = []
    for i in range(len(df)):
        results.append({
            "prediction": pred_lbl[i],
            "probabilities": {str(c): float(proba[i, j]) for j, c in enumerate(label_encoder.classes_)}
        })
    return {"count": len(results), "results": results}
