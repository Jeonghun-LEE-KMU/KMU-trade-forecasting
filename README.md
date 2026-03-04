# 제3회 국민대 공모전 : 수출입 무역량 예측

> **DACON 공모전** | 2025.11 | 최종 스코어: **0.4058**

## 대회 개요

| 항목 | 내용 |
|------|------|
| 주최 | 국민대학교 |
| 플랫폼 | DACON |
| 주제 | HS4 코드별 월별 수출입 무역량(value) 예측 |
| 평가 지표 | SMAPE (Symmetric Mean Absolute Percentage Error) |
| 최종 스코어 | **0.4058** |

## 문제 정의

HS4(국제통일상품분류체계 4자리) 코드별 월별 수출입 거래량(`value`)을 예측하는 시계열 회귀 문제.

- **핵심 난이도**: 품목 간 공행성(Co-movement) 관계를 활용한 예측 정확도 향상
- 전통적 시계열 모델이 아닌 **Cross-item 의존성** 파악이 핵심
- 다양한 피처 엔지니어링 + 앙상블 전략

## 사용 기술 스택

- **모델**: CatBoost, XGBoost, AutoGluon
- **라이브러리**: `catboost`, `xgboost`, `autogluon`, `optuna`, `pandas`, `numpy`, `sklearn`
- **핵심 기법**: 공행성(Co-movement) 탐지, Pearson 상관계수, DTW (Dynamic Time Warping)
- **최적화**: Optuna 하이퍼파라미터 튜닝

---

## 핵심 방법론: 공행성(Co-movement) 기반 피처 생성

### 공행성이란?

서로 다른 품목(item_id) 간에 시계열 패턴이 유사하게 움직이는 관계. A 품목이 오르면 B 품목도 일정 시차(lag)를 두고 오르는 현상.

### 공행성 탐지 파이프라인

```
1. 품목별 월별 무역량 피벗 테이블 생성
      ↓
2. 각 (A, B) 품목 쌍에 대해 lag = 1~max_lag Pearson 상관계수 계산
      ↓
3. |corr| ≥ threshold인 쌍 → 공행성 있다고 판단
      ↓
4. 3개 필터 (Pearson + DTW + 가중치) 조합으로 노이즈 쌍 제거
      ↓
5. A의 lag-k 이전 값을 B의 피처로 추가
```

### 피처 엔지니어링

| 피처 유형 | 내용 |
|-----------|------|
| 전월 대비 성장률 | MoM (Month-over-Month) 변화율 |
| 전년 동월 대비 성장률 | YoY (Year-over-Year) 변화율 |
| 단가 피처 | value / 수량 (단위 가격) |
| HS2 코드 | HS4 앞 2자리 카테고리화 |
| 공행성 lag 피처 | 연관 품목의 이전 시점 값 |

---

## 실험 흐름

```
BASELINE_p_corr          →  Pearson 상관계수 기반 공행성
BASELINE_p_corr_dtw      →  DTW 추가 (패턴 유사도 강화)
BASELINE_corr_filter     →  상관계수 필터 최적화
Basic_Best_Baseline      →  성능 향상 피처 자동 선택
BEST_weight_filtering    →  공행성 판단 가중치 + 임계값 튜닝 → 0.4058
```

### 최종 모델 설정 (BEST_weight_filtering)

| 설정 | 값 |
|------|----|
| 모델 | CatBoostRegressor |
| 공행성 임계값 | 0.35 |
| Dropout 비율 | 0.04 |
| 피처 선택 | 성능 향상 피처만 자동 선택 |
| Oversampling | 제거 (성능 저하 확인) |
| 최종 스코어 | **0.4058459231** |

---

## 프로젝트 구조

```
제3회국민대공모전/
├── notebooks/
│   ├── BASELINE_p_corr_dtw.ipynb      # DTW 기반 공행성 탐지 베이스라인
│   ├── Basic_Best_Baseline.ipynb      # 피처 자동 선택 버전
│   ├── BEST_weight_filtering.ipynb    # 최종 제출 코드 (Score: 0.4058)
│   └── Model_Compare.ipynb            # 모델 비교 실험
├── results/                           # 제출 파일 저장
└── README.md
```

> 원본 데이터, submission CSV, pkl 모델 파일은 용량 문제로 제외.

---

## 주요 인사이트 및 배운 점

1. **공행성(Co-movement) 활용**: 단일 품목 시계열만 보는 것이 아니라 연관 품목 간 lag 관계를 피처로 만들면 예측 성능이 유의미하게 향상됨.

2. **DTW vs Pearson**: Pearson 상관계수만 사용 시 직선적 유사도만 파악 가능. DTW를 추가하면 비선형 패턴 유사도까지 포착 가능.

3. **Oversampling 효과 없음**: 희소 품목에 대한 Oversampling을 시도했으나 SMAPE 기준 오히려 성능 저하. 데이터 분포 왜곡 우려.

4. **피처 자동 선택**: 무작위로 피처를 추가하는 것보다 검증 성능이 올라가는 피처만 자동으로 선택하는 greedy selection이 효과적.

5. **Optuna 튜닝**: CatBoost 하이퍼파라미터 자동 탐색으로 기본값 대비 성능 향상 확인.

---

## 참고 자료

- [DACON 공모전 페이지](https://dacon.io)
- [CatBoost 공식 문서](https://catboost.ai/docs/)
- [Dynamic Time Warping (DTW)](https://en.wikipedia.org/wiki/Dynamic_time_warping)
- [Optuna](https://optuna.org/)
