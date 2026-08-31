# 멀티모달 뇌영상 기반 TMS 전·후 변화 분석 및 반응자 예측

**경희대학교 소프트웨어융합 캡스톤디자인**

## 프로젝트 개요

본 프로젝트는 **TMS 치료 전·후의 멀티모달 뇌영상 데이터를 분석**하고, 치료 반응자(Responder)를 정의 및 예측하는 것을 목표로 한다.

현재까지는 멀티모달 데이터 중 **구조적 뇌영상(Morphometry)**과 **기능적 연결성(Functional Connectivity, FC)**을 중심으로 분석 및 예측 모델을 구축하였다.

```text
[T0 뇌영상 + 임상 정보] → TMS 후 갈망(VAS) 예측
```

## 주요 내용

* **Morphometry** 기반 뇌 구조 지표 분석
* **Functional Connectivity (FC)** 기반 뇌 기능 연결성 분석
* TMS 전·후 변화 및 치료 반응과의 관계 분석
* 치료 전 뇌영상 및 임상 정보를 활용한 **TMS 반응 예측 모델 구축**
* Nested Cross-Validation 기반 모델 성능 평가
* SHAP 등을 활용한 주요 feature의 기여도 해석
* 향후 **Responder 정의 및 이진 분류 모델** 구축

## Dataset

**SUDMEX-TMS** (OpenNeuro `ds003037`)

* 5-Hz rTMS 이중맹검 RCT
* Active 25명 / Sham 20명
* **Morphometry:** 152 features
* **Functional Connectivity:** 84 nodes, 3,486 edges

## 현재까지의 결과

* Active군에서는 **치료 전 VAS(`vas_t0`)를 활용한 치료 후 VAS(`vas_t1`) 예측이 가능**했음
* 뇌영상 feature를 추가하더라도 현재 분석에서는 **Clinical Only 대비 뚜렷한 증분 예측 성능이 확인되지 않음**
* Morphometry 분석을 완료했으며, **FC 기반 예측 분석을 진행 중**
* 향후 `vas_t0`의 영향을 통제한 뇌영상 기반 분석을 통해 뇌영상의 추가적인 설명력을 검증할 예정

## 향후 계획

1. Morphometry와 FC 결과 비교
2. `vas_t0` 통제 후 뇌영상 기반 예측 분석
3. Active vs Sham 반응 차이 검증
4. **TMS Responder 정의 및 이진 분류**
5. 주요 뇌영상 feature의 기여도 및 해석
