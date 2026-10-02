# -*- coding: utf-8 -*-
"""bctpy로 MSN / FC node strength를 뽑는다.

설치: pip install bctpy      (의존성은 numpy, scipy뿐)

bctpy의 strength 함수는 매우 단순하다. GitHub aestrivex/bctpy 기준으로
bct/algorithms/degree.py 에 들어 있고, 소스가 이게 전부다.

    def strengths_und(CIJ):
        return np.sum(CIJ, axis=0)

    def strengths_und_sign(W):
        W = W.copy()
        np.fill_diagonal(W, 0)
        Spos = np.sum(W * (W > 0), axis=0)
        Sneg = np.sum(W * (W < 0), axis=0)
        vpos = np.sum(W[W > 0]);  vneg = np.sum(W[W < 0])
        return Spos, Sneg, vpos, vneg

즉 strength는 "행렬 한 행의 합"이다. 우리가 build_msn_dataset.py / build_fc_dataset.py에서
직접 계산한 값(행 평균 = 합 / 67)과 상수배만 다르다. 이 스크립트는 그 일치를 검증한 뒤
bctpy 출력을 CSV로 남긴다.

★ 주의 1 — strengths_und는 대각을 지우지 않는다
   np.sum을 그냥 하므로, 대각이 1인 상관행렬을 넣으면 모든 노드에 1이 더해진다.
   그래서 아래에서 대각을 0으로 두고 행렬을 만든다. (strengths_und_sign은 내부에서 지운다)

★ 주의 2 — 음수
   strength는 부호 있는 값을 그대로 더한다. 계산은 되지만 양수와 음수가 서로 상쇄된다.
   MSN은 edge의 절반 가까이가 음수라 "합"이 0 근처의 순(net)값이 된다.
   그래서 BCT는 부호가 섞인 행렬에는 strengths_und_sign(양/음을 분리)을 권한다.
   이 스크립트는 둘 다 저장해 두고 노트북에서 고르게 한다.

출력 (bct_strength/ 폴더, 기존 CSV와 같은 레이아웃: 앞 7열 메타 + 뇌 피처)
    msn_bct_strength_68.csv         68개  = strengths_und / 67   (기존 msn_strength_68과 동일해야 함)
    msn_bct_strength_signed_68.csv  136개 = Spos 68 + Sneg 68
    fc_bct_strength_68.csv          68개
    fc_bct_strength_signed_68.csv   136개
"""

import os

import numpy as np
import pandas as pd

try:
    import bct
except ImportError:
    raise SystemExit("bctpy가 없습니다.  pip install bctpy  로 설치하세요.")

N_ROI = 68
META = 7          # rid, treatment_group, vas_t0, vas_t1, age, sex, (eTIV | mean_fd)
OUT_DIR = "bct_strength"

JOBS = [
    # (이름, edge CSV, strength CSV(검증용), edge 컬럼 접미사)
    ("msn", "msn_full_68x68.csv", "msn_strength_68.csv", "_msn"),
    ("fc",  "fc_full_68x68.csv",  "fc_strength_68.csv",  ""),
]


def load(path):
    d = pd.read_csv(path, skipinitialspace=True)
    d.columns = d.columns.str.strip()
    return d


def parse_edges(edge_cols, suffix):
    """edge 컬럼명 -> (i, j) 인덱스 쌍과 ROI 순서.

    컬럼명은 'lh_bankssts__lh_cuneus' (+ 접미사) 형식이다.
    ROI 순서는 strength CSV와 맞춰야 하므로 바깥에서 받은 순서를 쓴다.
    """
    pairs = []
    for c in edge_cols:
        name = c[:-len(suffix)] if suffix and c.endswith(suffix) else c
        a, b = name.split("__")
        pairs.append((a, b))
    return pairs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for tag, edge_csv, str_csv, suffix in JOBS:
        print(f"\n{'=' * 70}\n{tag.upper()}  ({edge_csv})")

        edge_df = load(edge_csv)
        str_df = load(str_csv)

        meta_cols = list(edge_df.columns[:META])
        edge_cols = list(edge_df.columns[META:])
        assert len(edge_cols) == N_ROI * (N_ROI - 1) // 2, len(edge_cols)

        # ROI 순서는 strength CSV 기준 (기존 노트북들과 동일한 순서 보장)
        str_suffix = "_msn" if tag == "msn" else "_fc"
        rois = [c[:-len(str_suffix)] for c in str_df.columns[META:]]
        assert len(rois) == N_ROI, len(rois)
        idx = {r: i for i, r in enumerate(rois)}

        pairs = parse_edges(edge_cols, suffix)
        ii = np.array([idx[a] for a, b in pairs])
        jj = np.array([idx[b] for a, b in pairs])

        S, Spos, Sneg = [], [], []
        for _, row in edge_df.iterrows():
            w = row[edge_cols].to_numpy(dtype=float)

            # 상삼각 -> 대칭 행렬. 대각은 0으로 둔다 (strengths_und가 대각을 안 지운다)
            W = np.zeros((N_ROI, N_ROI))
            W[ii, jj] = w
            W = W + W.T
            assert np.allclose(np.diag(W), 0)

            S.append(bct.strengths_und(W))                 # 부호 그대로 합
            sp, sn, _, _ = bct.strengths_und_sign(W)       # 양/음 분리
            Spos.append(sp)
            Sneg.append(sn)

        S = np.array(S)
        Spos, Sneg = np.array(Spos), np.array(Sneg)

        # ---- 검증: bct 합 / 67 == 기존 strength CSV ----
        ours = str_df[[f"{r}{str_suffix}" for r in rois]].to_numpy(dtype=float)
        mine = S / (N_ROI - 1)
        same_rid = edge_df["rid"].to_numpy() == str_df["rid"].to_numpy()
        assert same_rid.all(), "두 CSV의 rid 순서가 다릅니다"
        diff = np.abs(mine - ours).max()
        print(f"  기존 strength와 최대 차이 : {diff:.2e}   {'일치' if diff < 1e-9 else '★불일치★'}")

        neg_ratio = (np.array([r[edge_cols].to_numpy(dtype=float) for _, r in edge_df.iterrows()]) < 0).mean()
        print(f"  edge 중 음수 비율         : {neg_ratio:.1%}")
        print(f"  strength(부호합) 평균     : {mine.mean():+.4f}")
        print(f"  Spos 평균 / Sneg 평균     : {(Spos / (N_ROI - 1)).mean():+.4f} / {(Sneg / (N_ROI - 1)).mean():+.4f}")

        # ---- 저장 ----
        meta = edge_df[meta_cols]

        out1 = pd.concat([meta,
                          pd.DataFrame(mine, columns=[f"{r}_bctstr" for r in rois])], axis=1)
        p1 = os.path.join(OUT_DIR, f"{tag}_bct_strength_68.csv")
        out1.to_csv(p1, index=False)

        out2 = pd.concat([meta,
                          pd.DataFrame(Spos / (N_ROI - 1), columns=[f"{r}_bctpos" for r in rois]),
                          pd.DataFrame(Sneg / (N_ROI - 1), columns=[f"{r}_bctneg" for r in rois])], axis=1)
        p2 = os.path.join(OUT_DIR, f"{tag}_bct_strength_signed_68.csv")
        out2.to_csv(p2, index=False)

        print(f"  저장: {p1}         {out1.shape[0]} x {out1.shape[1]}")
        print(f"  저장: {p2}  {out2.shape[0]} x {out2.shape[1]}")

    print("\n완료. 두 CSV 모두 앞 7열이 메타라 기존 노트북처럼 index 7부터 뇌 피처다.")


if __name__ == "__main__":
    main()
