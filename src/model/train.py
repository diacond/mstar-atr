"""
HOG 특징 기반 SAR 표적 10-클래스 분류기 학습·평가.

평가는 무작위 분할 대신 MSTAR SOC의 공식 분할(학습 17°, 시험 15° 관측각)을
그대로 쓴다. 학습 때 보지 못한 촬영 조건에서 시험하기 위해서다.

분류기: RandomForest는 정확도 60%에 그쳤다. HOG는 1764차원의 연속형
피처라 축 기준 분할에 의존하는 트리 모델과 잘 맞지 않는다.
StandardScaler + RBF SVM으로 87%를 얻어 이 조합을 쓴다.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_split(processed_dir: Path, split: str):
    data = np.load(processed_dir / f"{split}_features.npz", allow_pickle=True)
    return data["X"], data["y"]


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
        ]
    )


def train_and_evaluate(processed_dir: Path):
    X_train, y_train = load_split(processed_dir, "train")
    X_test, y_test = load_split(processed_dir, "test")

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    classes = sorted(set(y_test) | set(y_train))
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    return model, classes, report, cm


if __name__ == "__main__":
    processed_dir = Path("data/processed")
    model, classes, report, cm = train_and_evaluate(processed_dir)

    print(f"전체 정확도(accuracy): {report['accuracy']:.4f}\n")
    print("클래스별 precision/recall/f1:")
    for cls in classes:
        m = report[cls]
        print(f"  {cls:10s} precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"f1={m['f1-score']:.3f} (n={int(m['support'])})")

    joblib.dump(model, processed_dir / "sar_classifier.joblib")
    joblib.dump(classes, processed_dir / "class_labels.joblib")

    (processed_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savetxt(processed_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(classes), comments="")

    print(f"\n모델 저장 -> {processed_dir / 'sar_classifier.joblib'}")
