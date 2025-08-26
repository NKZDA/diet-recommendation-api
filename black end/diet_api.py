# -*- coding: utf-8 -*-
"""
Diet Recommendation API (FastAPI)
รันด้วย:
    python -m uvicorn --app-dir "black end" diet_api:app --reload --port 8000
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import io, json, joblib, pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------- โหลดโมเดล + Encoder ----------
def _load_first(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return joblib.load(p)
    raise FileNotFoundError(f"❌ ไม่พบไฟล์: {paths}")

pipe = _load_first([
    Path(__file__).parent / "diet_model.joblib"
])
label_encoder = _load_first([
    Path(__file__).parent / "label_encoder.joblib"
])

# ---------- โหลด Schema ----------
schema_path = Path(__file__).parent / "model_schema.json"
if not schema_path.exists():
    raise FileNotFoundError("❌ ไม่พบ model_schema.json — โปรดเซฟ schema จาก Notebook ก่อน")
schema = json.loads(schema_path.read_text(encoding="utf-8"))
FEATURES = schema["features"]
CLASSES = schema["classes"]

def _align_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = FEATURES["categorical"] + FEATURES["numeric"]
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[cols]
    for c in FEATURES["numeric"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in FEATURES["categorical"]:
        df[c] = df[c].astype(str)
    return df

# ---------- FastAPI ----------
app = FastAPI(title="Diet Recommendation API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

class PredictBody(BaseModel):
    record: Optional[Dict[str, Any]] = None
    records: Optional[List[Dict[str, Any]]] = None
    return_proba: bool = True

@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASSES, "n_classes": len(CLASSES)}

@app.get("/schema")
def schema_endpoint():
    return {"features": FEATURES, "classes": CLASSES}

@app.post("/predict")
def predict(body: PredictBody):
    if body.records is None and body.record is None:
        return {"error": "ต้องส่ง 'record' หรือ 'records'"}
    rows = body.records if body.records is not None else [body.record]
    df = _align_df(pd.DataFrame(rows))
    proba = pipe.predict_proba(df)
    idx = proba.argmax(axis=1)
    lbl = label_encoder.inverse_transform(idx)
    out = []
    for i in range(len(df)):
        item = {"predicted_label": str(lbl[i]), "predicted_index": int(idx[i])}
        if body.return_proba:
            item["probabilities"] = {str(c): float(proba[i, j]) for j, c in enumerate(label_encoder.classes_)}
        out.append(item)
    return {"count": len(out), "predictions": out}

@app.post("/predict-csv")
def predict_csv(file: UploadFile = File(...), return_proba: bool = True):
    content = file.file.read()
    df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    df = _align_df(df)
    proba = pipe.predict_proba(df)
    idx = proba.argmax(axis=1)
    lbl = label_encoder.inverse_transform(idx)
    out = []
    for i in range(len(df)):
        row = {"predicted_label": str(lbl[i]), "predicted_index": int(idx[i])}
        if return_proba:
            row["probabilities"] = {str(c): float(proba[i, j]) for j, c in enumerate(label_encoder.classes_)}
        out.append(row)
    return {"count": len(out), "predictions": out}

