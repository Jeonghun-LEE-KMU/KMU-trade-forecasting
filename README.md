<div align="center">

# 📈 제3회 국민대학교 AI빅데이터 분석 경진대회

**무역 품목 간 공행성(Comovement) 쌍 판별 및 후행 품목 무역량 예측**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat-square&logo=catboost&logoColor=black)](https://catboost.ai/)
[![Optuna](https://img.shields.io/badge/Optuna-blue?style=flat-square)](https://optuna.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)

`DACON` · `국민대학교 경영대학원 × KOAMI` · `2025.11` · `960팀 참가` · **78위 / 960** · *SMAPE: 0.4058**

</div>

---

## 📌 Key Highlights

| | |
|:---|:---|
| 🏆 **결과** | **78위 / 960팀** — 최종 SMAPE **0.4058** |
| 🎯 **핵심 과제** | 100개 수입 품목에서 공행성(Comovement) 쌍을 **자체 설계한 방법론**으로 판별 |
| ⚡ **접근 방식** | Pearson 상관계수 + DTW(Dynamic Time Warping) 기반 3중 필터 파이프라인 |
| 📊 **모델** | CatBoost (Ordered Boosting으로 시계열 target leakage 방지) |
| 🔧 **피처 선택** | Greedy Forward Selection으로 과적합 억제 |

---

## 🔍 과제 정의

무역 데이터에서 품목 간 구조적 관계를 탐색하고, 예측 기반 의사결정 지원 도구로 활용 가능한 AI 모델을 개발하는 대회입니다.

**두 가지 과제:**

1. **공행성(Comovement) 쌍 판별** — 100개 품목(item_id)의 과거 거래 데이터에서, 시간적으로 유사하게 움직이는 공행성 쌍(선행 → 후행)을 식별
2. **후행 품목 무역량 예측** — 선행 품목의 흐름을 기반으로, 후행 품목의 다음 달 총 무역량(value)을 예측

> 대회는 공행성 쌍 판별 방법론을 규정하지 않음 — **판별 방법 자체를 설계하는 것이 핵심 난이도**

**평가 지표 — SMAPE:**

$$\text{SMAPE} = \frac{100\%}{n} \sum_{t=1}^{n} \frac{|F_t - A_t|}{(|A_t| + |F_t|)/2}$$

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Detection["🔍 공행성 쌍 판별"]
        A["📇 item_id별 월별 value\n시계열 피벗"] --> B["📐 Pearson 상관계수\n(lag=1~max_lag)"]
