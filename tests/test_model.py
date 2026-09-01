from pathlib import Path

import pytest

from src.model.train import train_and_evaluate

PROCESSED_DIR = Path("data/processed")


@pytest.fixture(scope="module")
def trained():
    """모듈 안에서 한 번만 학습해서 재사용 (SVM 학습이 몇십 초 걸려서)."""
    return train_and_evaluate(PROCESSED_DIR)


def test_model_beats_random_baseline(trained):
    """10-클래스 랜덤 추측은 10% 정확도. 이보다 훨씬 잘해야 의미가 있다.

    실제 홀드아웃(15도 관측각) 정확도는 87% 안팎으로 나오는데, 회귀
    테스트로는 너무 타이트하게 잡지 않고 60%를 최소선으로 뒀다 - 향후
    피처/모델을 바꿔도 "완전히 망가졌는지"만 잡아내는 안전망 역할.
    """
    _, _, report, _ = trained
    assert report["accuracy"] > 0.6


def test_confusion_matrix_shape_matches_class_count(trained):
    _, classes, _, cm = trained
    assert cm.shape == (len(classes), len(classes))
    assert len(classes) == 10
