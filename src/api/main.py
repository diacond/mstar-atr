"""
SAR 표적 분류 API.

이미지를 업로드하면 10개 차량 클래스 중 예측 클래스와 클래스별 확률을
반환한다. "확신도가 낮으면 재검토" 같은 판단 정책은 호출하는 쪽에서 정하도록
확률 분포를 그대로 노출한다.
"""

from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from src.pipeline.build_features import extract_features_from_image

MODEL_PATH = Path("data/processed/sar_classifier.joblib")
LABELS_PATH = Path("data/processed/class_labels.joblib")

app = FastAPI(title="SAR Target Recognition API", version="0.1.0")

_model = None
_classes = None


def get_model():
    global _model, _classes
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="모델이 없습니다. src/model/train.py를 먼저 실행하세요.")
        _model = joblib.load(MODEL_PATH)
        _classes = joblib.load(LABELS_PATH)
    return _model, _classes


class ClassificationResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/v1/sar/classify", response_model=ClassificationResponse)
async def classify(file: UploadFile = File(...)):
    model, classes = get_model()

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"이미지를 열 수 없습니다: {exc}") from exc

    features = extract_features_from_image(image).reshape(1, -1)
    probabilities = model.predict_proba(features)[0]
    predicted_idx = int(np.argmax(probabilities))

    return ClassificationResponse(
        predicted_class=model.classes_[predicted_idx],
        confidence=round(float(probabilities[predicted_idx]), 4),
        class_probabilities={
            cls: round(float(p), 4) for cls, p in zip(model.classes_, probabilities)
        },
    )
