"""
HOG 특징으로 SAR 표적 10-클래스 분류기 학습 및 평가.

평가에 랜덤 train/test 분할을 쓰지 않는다 — MSTAR SOC 데이터셋 자체가
이미 "train은 17도 관측각, test는 15도 관측각"으로 서로 다른 촬영
조건을 갖도록 공식적으로 나뉘어 있어서, 이 분할을 그대로 쓰는 게 표준
관례이자 가장 정직한 평가다(배터리 프로젝트에서 Leave-One-Battery-Out을
쓴 것과 같은 이유: "학습 때 보지 못한 조건"에서 시험해야 한다).

분류기로 RandomForest를 먼저 시도했는데 정확도 60%에 그쳤다. HOG는
고차원(1764차원)이고 각 차원의 스케일이 제각각인데, 트리 기반 모델은
이런 고차원 연속형 피처의 미세한 방향 차이를 잘 못 살린다. 반면 SVM은
피처를 표준화(StandardScaler)한 뒤 RBF 커널로 넣었더니 87%까지 올라가서
이걸로 확정했다 — "HOG면 무조건 SVM"이라는 관례가 왜 있는지 직접
겪어본 셈이다.
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
