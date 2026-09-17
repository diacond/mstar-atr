# SAR Target Recognition (MSTAR)

합성개구레이더(SAR) 영상에서 군용 차량 10종을 자동으로 식별하는 표적 인식(ATR) 파이프라인. 레이더 정찰 영상의 1차 판독을 자동화하는 서비스를 작은 규모로 구현했다.

> 쉬운 설명은 [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md)에 정리했다.

## 문제 정의

SAR은 날씨·주야와 관계없이 촬영할 수 있지만, 색 정보가 없고 촬영 각도에 따라 같은 물체도 다르게 보여 판독이 어렵다. SAR 칩(표적 부분을 잘라낸 영상) 한 장을 보고 10종 차량 중 무엇인지 맞히는 표준 벤치마크 문제를 다룬다.

## 데이터

[MSTAR Public Targets](https://www.sdms.afrl.af.mil/index.php?collection=mstar&page=targets)의 10-클래스 SOC(Standard Operating Condition) 분할. 그레이스케일 SAR 영상을 JPEG로 변환한 공개본을 쓴다.

| 분류 | 클래스 |
| --- | --- |
| 전차 | T62, T72 |
| 장갑차 | BMP2, BRDM_2, BTR60, BTR70 |
| 자주포 | 2S1 |
| 대공차량 | ZSU_23_4 |
| 트럭 | ZIL131 |
| 불도저 | D7 |

학습은 17°, 시험은 15° 관측각으로 촬영한 서로 다른 영상이다(학습 2,746장 / 시험 2,425장). 다른 조건에서 찍은 영상으로 시험하는 구조가 데이터셋에 이미 들어 있다.

## 파이프라인

```
data/raw/{train,test}/<class>/*.jpeg
   │  src/pipeline/build_features.py   중앙 64x64 크롭 + HOG
   ▼
{train,test}_features.npz
   │  src/model/train.py               StandardScaler + RBF SVM
   ▼
sar_classifier.joblib
   │  src/api/main.py
   ▼
POST /v1/sar/classify (이미지 업로드) → { predicted_class, confidence, class_probabilities }
```

## 주요 설계 판단

- **CNN 대신 HOG + SVM으로 시작:** 데이터가 5천 장 남짓이고, 어떤 클래스가 왜 헷갈리는지 특징 단위로 분석하기 쉬워서 고전적 방법으로 베이스라인을 먼저 잡았다.
- **크롭 크기 64:** 128x128로 자르면 정확도가 78%였다. SAR 영상은 표적 주변에 배경 클러터와 스페클 노이즈가 넓게 깔려 있어, 크게 자를수록 노이즈 비중이 커진다.

  | 크롭 | 64 | 80 | 96 | 128 |
  | --- | --- | --- | --- | --- |
  | 정확도 | **87%** | 84% | 81% | 78% |

- **RandomForest 대신 SVM:** 같은 HOG 특징으로 RandomForest는 60%에 그쳤다. 1,764차원 연속형 피처에서는 축 기준 분할에 의존하는 트리 모델이 불리하다. 표준화 후 RBF SVM을 쓰자 87%가 나왔다.

## 결과

테스트(15° 관측각) 정확도 **86.7%**

| 클래스 | Precision | Recall | F1 | 샘플 수 |
|---|---|---|---|---|
| 2S1 | 0.78 | 0.80 | 0.79 | 274 |
| BMP2 | 0.93 | 0.72 | 0.81 | 195 |
| BRDM_2 | 0.96 | 0.85 | 0.90 | 274 |
| BTR60 | 0.80 | 0.82 | 0.81 | 195 |
| BTR70 | 0.86 | 0.84 | 0.85 | 196 |
| D7 | 0.95 | 0.96 | 0.96 | 274 |
| T62 | 0.79 | 0.88 | 0.83 | 273 |
| T72 | 0.85 | 0.82 | 0.83 | 196 |
| ZIL131 | 0.88 | 0.94 | 0.91 | 274 |
| ZSU_23_4 | 0.90 | 0.97 | 0.93 | 274 |

### 혼동 분석

혼동행렬(`data/processed/confusion_matrix.csv`)에서 가장 많이 헷갈린 쌍은 **BTR60 ↔ BTR70**, **2S1 ↔ T62**다.

- BTR60과 BTR70은 외형이 거의 같은 8x8 바퀴형 장갑차로, SAR ATR 연구에서도 구분이 어려운 쌍으로 알려져 있다.
- 2S1(자주포)과 T62(전차)는 크기와 형태가 비슷한 궤도형 차량이다.
- SAR은 형태 정보만으로 구분해야 해서, 실루엣이 비슷한 차량끼리는 구조적으로 헷갈리기 쉽다.

## 실행

```bash
pip install -r requirements.txt

bash scripts/download_data.sh            # 데이터 준비
python src/pipeline/build_features.py    # HOG 특징 추출
python src/model/train.py                # 학습·평가
uvicorn src.api.main:app --reload        # API 서버
pytest                                   # 테스트
```

## CI

GitHub Actions에서 데이터 다운로드 → 특징 추출 → 학습 → 테스트를 매번 처음부터 실행한다.

## 남은 과제

- CNN과 정확도·학습 시간 비교
- BTR60/BTR70처럼 헷갈리는 쌍을 여러 관측각 영상으로 개선할 수 있는지 검토
- Docker 컨테이너화
- 확신도가 낮은 예측을 재검토 대상으로 분류하는 후속 처리
