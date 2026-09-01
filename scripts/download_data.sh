#!/usr/bin/env bash
# MSTAR Public Targets (10-class SOC 표준 분할, JPEG 변환본)을 내려받는다.
#
# 원본은 Sandia National Labs / AFRL이 공개한 SAR(합성개구레이더) 표적
# 인식 연구용 데이터셋(MSTAR Public Targets)이다. 여기서는 그레이스케일
# JPEG로 변환해 공개 배포 중인 미러(GitHub)에서 받는다.
set -euo pipefail

RAW_DIR="$(dirname "$0")/../data/raw"
mkdir -p "$RAW_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

git clone --depth 1 \
  https://github.com/hunterlew/mstar_with_machine_learning.git \
  "$TMP_DIR/repo"

cp -r "$TMP_DIR/repo/MSTAR-10/train" "$RAW_DIR/train"
cp -r "$TMP_DIR/repo/MSTAR-10/test" "$RAW_DIR/test"

echo "다운로드 완료 -> $RAW_DIR"
find "$RAW_DIR" -type f | wc -l
echo "개 이미지 파일 준비됨"
