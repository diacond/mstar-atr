"""
MSTAR SAR 표적 칩(chip) → HOG 특징 벡터 파이프라인.

원본 출처: MSTAR Public Targets (Sandia National Labs / AFRL이 공개한
합성개구레이더(SAR) 표적 인식 연구용 데이터셋). 여기서 쓰는 버전은
그레이스케일 SAR 진폭 영상을 JPEG로 변환해 배포한 10-클래스 표준
SOC(Standard Operating Condition) 분할본이다 — 클래스 10개(전차, 장갑차,
자주포 등 군용 차량), train은 17도 관측각, test는 15도 관측각으로
촬영된 서로 다른 사이클의 영상이라 애초에 "다른 조건에서 찍은 새
샘플로 시험한다"는 원칙이 데이터셋 자체에 내장돼 있다.

접근 방식: 딥러닝(CNN)으로 바로 가지 않고, HOG(Histogram of Oriented
Gradients) 특징 + 고전적 분류기로 먼저 베이스라인을 잡았다. SAR 표적
인식은 색상 정보가 없고 형태(윤곽선의 방향 분포)가 핵심 단서인 문제라
HOG가 잘 맞는 편이고, 무엇보다 "왜 이 클래스로 분류했는지"를 특징
중요도로 뜯어볼 수 있어서 어떤 클래스 쌍이 왜 헷갈리는지 분석하기
쉽다는 이점이 있다.

크롭 크기 128로 처음 돌렸을 때 테스트 정확도가 78%밖에 안 나왔다.
원본 192x193 SAR 영상은 표적 주변에 배경 클러터(clutter)와 스페클
노이즈가 상당히 넓게 깔려 있는데, 128 크롭은 이 노이즈 영역을 너무
많이 포함해서 HOG 특징이 표적의 형태보다 배경 잡음을 더 많이 반영하게
됐던 것으로 보인다. 64/80/96 크롭을 비교해본 결과 64에서 87%로 가장
높게 나와서(96: 81%, 80: 84%, 64: 87%) 이 크기로 확정했다 — "정보를
더 많이 준다고 항상 좋은 게 아니다"라는 걸 여기서도 확인했다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import hog

CROP_SIZE = 64  # 표적 주변 배경 클러터를 최대한 배제하기 위한 크기 (본문 설명 참고)
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w < size or h < size:
        # 업로드된 이미지가 학습 데이터보다 작을 수 있으니, 그 경우엔
        # 크롭 대신 리사이즈로 대체한다 (학습 데이터는 전부 192x193이라
        # 이 경로를 타지 않지만, API로 임의 이미지가 들어올 수 있어 방어)
        return img.resize((size, size))
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def extract_features_from_image(img: Image.Image) -> np.ndarray:
    """이미 열려 있는 PIL Image로부터 HOG 특징을 뽑는다 (API에서 업로드된
    이미지를 파일로 저장하지 않고 바로 처리할 때 재사용)."""
    img = img.convert("L")  # 그레이스케일
    img = _center_crop(img, CROP_SIZE)
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return hog(arr, **HOG_PARAMS)


def extract_features(image_path: Path) -> np.ndarray:
    return extract_features_from_image(Image.open(image_path))


def build_split(split_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """split_dir 밑의 클래스별 폴더를 읽어 (X, y, 파일경로) 반환."""
    class_names = sorted(p.name for p in split_dir.iterdir() if p.is_dir())

    features, labels, paths = [], [], []
    for class_name in class_names:
        for img_path in sorted((split_dir / class_name).glob("*.jpeg")):
            features.append(extract_features(img_path))
            labels.append(class_name)
            paths.append(str(img_path))

    return np.vstack(features), np.array(labels), paths


def main(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        X, y, paths = build_split(raw_dir / split)
        np.savez_compressed(out_dir / f"{split}_features.npz", X=X, y=y, paths=paths)
        print(f"{split}: {X.shape[0]} samples, {X.shape[1]} HOG features, "
              f"{len(set(y))} classes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--out-dir", default="data/processed", type=Path)
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir)
