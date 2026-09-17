"""
MSTAR SAR 표적 칩 → HOG 특징 벡터 파이프라인.

데이터: MSTAR Public Targets(Sandia National Labs / AFRL)의 10-클래스 SOC
분할을 JPEG로 변환한 공개본. 학습은 17°, 시험은 15° 관측각으로 촬영된
서로 다른 영상이다.

접근: CNN 대신 HOG + 고전적 분류기로 베이스라인을 잡았다. SAR 표적은 색
정보가 없고 윤곽선의 방향 분포가 핵심 단서라 HOG와 잘 맞고, 어떤 클래스가
왜 헷갈리는지 분석하기도 쉽다.

크롭 크기: 128에서는 정확도가 78%였다. 원본(192x193)은 표적 주변에 배경
클러터와 스페클 노이즈가 넓게 깔려 있어, 크게 자를수록 노이즈 비중이
커진다. 비교 결과(128: 78%, 96: 81%, 80: 84%, 64: 87%) 64로 정했다.
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
