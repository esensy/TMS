# -*- coding: utf-8 -*-
"""
MSN(Morphometric Similarity Network, Seidlitz et al. 2018)을 DK 68 cortical ROI로 만든다.
FC의 cortical 68 버전(fc_strength_68.csv / fc_full_68x68.csv)과 ROI가 순서까지 1:1로 대응한다.

출력 1: msn_strength_68.csv   (45 x 75)    ROI별 regional MS 68개
출력 2: msn_full_68x68.csv    (45 x 2285)  상삼각 edge 2278개

둘 다 앞 7개 컬럼이 rid, treatment_group, vas_t0, vas_t1, age, sex, eTIV라
구조 CSV(structural_full_68_68_16_edited.csv)와 똑같이 index 7부터 뇌 피처가 시작한다.

만드는 법 (피험자마다 따로)
  1) FreeSurfer aparc(DK) 8개 measure를 모아 68 ROI x 8 행렬 X를 만든다.
       ThickAvg ThickStd SurfArea GrayVol MeanCurv GausCurv FoldInd CurvInd
  2) X의 각 열(measure)을 68 ROI에 걸쳐 z-score 한다.
     단위가 mm, mm^2, mm^3, 1/mm로 제각각이라 그대로 두면 숫자가 큰 GrayVol이
     상관을 독점한다. z-score 후에는 "이 ROI가 이 사람 뇌 안에서 상대적으로
     두꺼운가/넓은가/굽었나"만 남는다.
  3) ROI 두 개의 8칸짜리 벡터끼리 Pearson r -> 68 x 68.
     FC는 ROI마다 시계열 300개(TR)로 상관을 내지만 구조 MRI는 ROI마다 measure당
     값이 1개뿐이라, 대신 measure 축 8개를 표본으로 빌려 쓴다. 이것이 MSN이다.
  4) regional MS = 자기 자신을 뺀 나머지 67개 r의 평균 (FC node strength와 같은 정의).

Fisher z를 하지 않는 이유
  FC는 arctanh(r)로 변환해 쓰지만 MSN은 r을 그대로 쓴다 (Seidlitz 원논문과 동일).
  Fisher z의 근거인 분산 안정화는 표본이 충분할 때 성립하는데, n=8에서는 |r|이 1에
  가까운 edge가 흔하고 arctanh가 이들을 크게 부풀려 strength 평균을 흔든다.

eTIV
  2)의 z-score가 피험자 안에서 이뤄지므로 머리 크기 같은 전역 스케일은 MSN에서
  이미 빠진다. 그래도 구조 CSV와 레이아웃을 맞추고 공변량으로 쓸 수 있게 7번째 컬럼에 둔다.

컬럼 이름
  strength는 lh_bankssts_msn (구조 _thick, FC _fc와 같은 접미사 규칙).
  edge는 lh_bankssts__lh_caudalanteriorcingulate_msn. FC edge(접미사 없음)와
  이름이 겹치면 두 표를 merge할 때 _x/_y가 붙으므로 _msn을 붙여 구분한다.

주의
  - 표본이 measure 8개뿐이고 그중 SurfArea/GrayVol/FoldInd/CurvInd는 서로 r~0.97이라
    유효 차원은 약 3이다. edge 하나하나는 불안정하므로 strength 버전을 먼저 쓴다.
  - measure 조합을 바꾸면 네트워크가 달라진다. 8개로 고정하고 사후에 바꾸지 않는다.
  - subcortical 16개는 표면 측정치(두께, 곡률 등)가 정의되지 않아 MSN에 들어갈 수 없다.
  - morphometry_DKTmapped_*.csv는 bankssts/frontalpole/temporalpole이 빠진 62 ROI라
    FC와 맞지 않는다. 반드시 DK 68 파일(morphometry_{measure}.csv)을 쓴다.
  - 구조 CSV 45명 전원이 들어간다. FC와 합치면 sub-001이 빠져 n=44가 된다.
"""

import os
import numpy as np
import pandas as pd

MORPH_DIR = r"D:\SUDMEX\Morphometry"
CLIN_CSV = "structural_full_68_68_16_edited.csv"
OUT_STRENGTH = "msn_strength_68.csv"
OUT_EDGES = "msn_full_68x68.csv"
SESSION = "ses-t0"

# 순서는 결과에 영향이 없다 (Pearson은 벡터 원소 순서를 타지 않는다).
MEASURES = ["ThickAvg", "ThickStd", "SurfArea", "GrayVol",
            "MeanCurv", "GausCurv", "FoldInd", "CurvInd"]

# 구조 CSV의 앞 7개 컬럼을 그대로 가져온다.
CLIN_COLS = ["rid", "treatment_group", "vas_t0", "vas_t1", "age", "sex", "eTIV"]


def load_measures():
    """8개 measure CSV -> {measure: DataFrame(index=rid, columns=ROI 68개)}.

    파일마다 ses-t0 행만 남기고 index를 'sub-002_ses-t0' -> 2로 바꾼다.
    8개 파일의 ROI 구성/순서가 하나라도 다르면 X의 행이 어긋나므로 멈춘다.
    """
    tables, ref_cols = {}, None
    for m in MEASURES:
        df = pd.read_csv(os.path.join(MORPH_DIR, f"morphometry_{m}.csv"), index_col=0)
        df.columns = df.columns.str.strip()
        df = df[df.index.str.endswith(f"_{SESSION}")]
        df.index = df.index.str.extract(r"sub-(\d+)_", expand=False).astype(int)
        if df.index.duplicated().any():
            raise RuntimeError(f"{m}: 같은 피험자가 {SESSION}에 두 번 이상 있음")
        if ref_cols is None:
            ref_cols = list(df.columns)
        elif list(df.columns) != ref_cols:
            raise RuntimeError(f"{m}: ROI 구성/순서가 {MEASURES[0]}와 다름")
        tables[m] = df
    return tables, ref_cols


def msn(X):
    """X: ROI x measure (68 x 8) -> ROI x ROI Pearson r (68 x 68).

    열마다 ROI에 걸쳐 z-score한 뒤 행(ROI의 8칸짜리 벡터)끼리 상관을 낸다.
    """
    sd = X.std(axis=0)
    if np.any(sd == 0):
        raise RuntimeError("모든 ROI에서 값이 같은 measure가 있어 z-score 불가")
    Z = (X - X.mean(axis=0)) / sd
    return np.corrcoef(Z)


def main():
    clin = pd.read_csv(CLIN_CSV, skipinitialspace=True)
    clin.columns = clin.columns.str.strip()
    # 구조 CSV의 _thick 컬럼 순서가 곧 FC 재정렬의 기준이므로 ROI 순서를 여기에 맞춘다.
    thick_rois = [c[:-len("_thick")] for c in clin.columns if c.endswith("_thick")]
    clin = clin[CLIN_COLS]

    tables, rois = load_measures()
    if rois != thick_rois:
        raise RuntimeError("morphometry ROI 순서가 구조 CSV의 _thick 순서와 다름")

    mats, rids = [], []
    for rid in clin["rid"]:
        if any(rid not in t.index for t in tables.values()):
            print(f"  sub-{rid:03d}: morphometry({SESSION}) 없음, 제외")
            continue
        X = np.column_stack([tables[m].loc[rid].values for m in MEASURES]).astype(float)
        if not np.all(np.isfinite(X)):
            raise RuntimeError(f"sub-{rid:03d}: morphometry에 비유한 값 존재")
        mats.append(msn(X))
        rids.append(rid)

    r = np.stack(mats)                                # (n_sub, 68, 68)
    n_sub, n_roi = r.shape[0], r.shape[1]
    diag = np.arange(n_roi)
    if not np.allclose(r, r.transpose(0, 2, 1)):
        raise RuntimeError("MSN 행렬이 대칭이 아님")
    if not np.allclose(r[:, diag, diag], 1.0):
        raise RuntimeError("MSN 대각이 1이 아님")

    # regional MS: 대각을 0으로 두고 나머지 67개 파트너의 평균
    rs = r.copy()
    rs[:, diag, diag] = 0.0
    strengths = rs.sum(axis=2) / (n_roi - 1)
    strength_names = [f"{n}_msn" for n in rois]

    # edge: 상삼각(대각 제외)만 취해 중복 제거
    iu = np.triu_indices(n_roi, k=1)
    edges = r[:, iu[0], iu[1]]
    edge_names = [f"{rois[i]}__{rois[j]}_msn" for i, j in zip(*iu)]

    def save(values, colnames, path, what):
        frame = pd.DataFrame(values, columns=colnames)
        frame.insert(0, "rid", rids)
        out = clin.merge(frame, on="rid", how="inner")
        assert list(out.columns[:7]) == CLIN_COLS, f"앞 7개 컬럼이 예상과 다름: {list(out.columns[:7])}"
        out.to_csv(path, index=False)
        print(f"  {path:22s} {out.shape[0]:3d} x {out.shape[1]:<5d} "
              f"(임상 7 + {what} {len(colnames)})")

    print(f"\n저장 ({n_sub}명, cortical ROI {n_roi}개, measure {len(MEASURES)}개):")
    save(strengths, strength_names, OUT_STRENGTH, "regional MS")
    save(edges, edge_names, OUT_EDGES, "edge")
    print(f"\n  strength 첫/끝 : {strength_names[0]} / {strength_names[-1]}")
    print(f"  edge 첫/끝     : {edge_names[0]} / {edge_names[-1]}")
    print(f"\n  edge 평균 {edges.mean():+.3f}, SD {edges.std():.3f}, "
          f"|r| > 0.9 비율 {np.mean(np.abs(edges) > 0.9):.1%}")
    print(f"  regional MS 범위 {strengths.min():+.3f} ~ {strengths.max():+.3f}")


if __name__ == "__main__":
    main()
