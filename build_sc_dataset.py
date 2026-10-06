# -*- coding: utf-8 -*-
"""
SC(84x84 DK, ses-t0)를 FC/MSN CSV와 같은 레이아웃으로 만든다.

SC는 상관이 아니라 streamline 개수다. DSI Studio(QSDR, 결정론적 추적 100만 개)로 만든
연결 행렬(.connectivity.mat)의 'number of tracts r2r'을 그대로 읽는다. 추적이나
행렬 생성은 여기서 하지 않는다 (run_sc_dk84_connectivity.sh가 이미 만들어 둔 것을 읽기만 한다).

연결을 세는 기준이 두 가지라 각각 따로 저장한다.
  pass : streamline이 두 영역을 모두 지나가면 센다 (DSI Studio 기본값).            <- 주 분석
  end  : streamline의 양 끝점이 두 영역 안에 있을 때만 센다 (Chu 2025, Qian 2026). <- 민감도 분석
pass는 streamline 하나가 여러 쌍에 중복으로 들어가고, end는 피질 라벨에 못 미쳐 멈춘
streamline을 버린다. 44명 기준 end의 총합은 pass의 9.4%, 밀도는 0.644 -> 0.416이다.
어느 쪽이 주 분석인지는 결과(vas_t1)를 보기 전에 pass로 정해 두었다.

출력 (sc_dk84/ 폴더, 기준마다 2개)
  sc_strength_84_{pass,end}.csv   (45 x 91)    ROI별 node strength 84개
  sc_full_84x84_{pass,end}.csv    (45 x 3493)  상삼각 edge 3486개 전부

앞 7개 컬럼은 rid, treatment_group, vas_t0, vas_t1, age, sex, eTIV라
구조 CSV(structural_full_68_68_16_edited.csv)와 똑같이 index 7부터 뇌 피처가 시작한다.
FC는 7번째 자리에 mean_fd를 넣었지만 그건 fMRI 촬영 중 움직임이라 SC와 무관하다.
SC strength는 뇌와 영역이 클수록 커지므로 Morph/MSN처럼 eTIV를 둔다.

값은 변환하지 않은 원래 count다.
  - SC 자체를 피처로 쓸 때는 노트북에서 log(1 + count)를 적용한다 (Tong 2026).
    strength는 "더한 뒤 log"다. edge마다 log를 씌워 더하면 다른 값이 된다.
  - SC-FC coupling은 Spearman이라 변환이 필요 없다 (순위만 쓴다).
원본을 남겨 두어야 나중에 다른 변환을 시험할 수 있다.

node strength 정의
  strength_i = sum_{j != i} count_ij   (bct.strengths_und)
  그 영역에 걸린 streamline 총수다. SC는 값이 전부 0 이상이라 FC/MSN과 달리 상쇄가 없다.
  FC/MSN CSV는 행 "평균"(합 / (R-1))을 저장했지만 여기는 "합"을 저장한다. 상수배 차이라
  표준화 후에는 같고, 합이어야 log(1 + x)가 작은 값에서도 자연스럽다.

ROI 순서는 fc_strength_68_16.csv와 같다 (피질 lh->rh 68개, 그 다음 피질하 lh->rh 16개).
edge 컬럼 순서도 fc_full_84x84_edited.csv와 같아서, k번째 SC edge와 k번째 FC edge가
같은 영역쌍이다. 이름만 '_sc' 접미사로 구분한다 (lh_bankssts__lh_cuneus_sc).

주의: SC는 45명 전원에게 있다. FC는 sub-001이 없어 44명이므로, SC-FC coupling처럼
두 모달리티가 다 필요한 분석은 rid=1을 빼고 44명이 된다.
"""

import glob
import os

import numpy as np
import pandas as pd

try:
    import bct
except ImportError:
    raise SystemExit("bctpy가 없습니다.  pip install bctpy  로 설치하세요.")

SC_ROOTS = {
    "pass": r"D:\SUDMEX\Connectivity_DK84_QSDR",
    "end":  r"D:\SUDMEX\Connectivity_DK84_QSDR_end",
}
CLIN_CSV = "structural_full_68_68_16_edited.csv"
FC_STRENGTH_CSV = "fc_strength_68_16.csv"       # ROI 순서 대조용
FC_EDGES_CSV = "fc_full_84x84_edited.csv"       # edge 순서 대조용
OUT_DIR = "sc_dk84"
SESSION = "ses-t0"
N_ROI = 84
COUNT_KEY = "number of tracts r2r"              # DSI Studio가 .mat에 쓰는 이름

FINAL_COLS = ["rid", "treatment_group", "vas_t0", "vas_t1", "age", "sex", "eTIV"]


def sc_mat(root, rid):
    sub = f"sub-{rid:03d}"
    hits = glob.glob(os.path.join(root, sub, SESSION, "*.connectivity.mat"))
    return hits[0] if hits else None


def load_counts(path):
    """.connectivity.mat -> (ROI 이름 84개, 84x84 count 행렬).

    'name'은 라벨을 줄바꿈으로 이어 붙인 문자 코드 배열이다.
    """
    from scipy.io import loadmat
    m = loadmat(path)
    names = [n for n in "".join(chr(int(c)) for c in m["name"].ravel()).split("\n") if n][:N_ROI]
    return names, m[COUNT_KEY].astype(float)


def build(tag, root, clin, fc_names, fc_edge_names):
    mats, rids, ref_names = [], [], None
    for rid in clin["rid"]:
        p = sc_mat(root, rid)
        if p is None:
            print(f"  sub-{rid:03d}: SC({tag}, {SESSION}) 없음, 제외")
            continue
        names, c = load_counts(p)
        if ref_names is None:
            ref_names = names
        elif names != ref_names:
            raise RuntimeError(f"sub-{rid:03d}: 라벨 순서/구성이 기준과 다름")
        if c.shape != (N_ROI, N_ROI):
            raise RuntimeError(f"sub-{rid:03d}: 행렬 크기 {c.shape}")
        if not np.allclose(c, c.T):
            raise RuntimeError(f"sub-{rid:03d}: 연결 행렬이 대칭이 아님")
        if (c < 0).any() or not np.allclose(c, np.round(c)):
            raise RuntimeError(f"sub-{rid:03d}: count가 음수이거나 정수가 아님")
        np.fill_diagonal(c, 0.0)                 # bct.strengths_und는 대각을 지우지 않는다
        mats.append(c)
        rids.append(rid)

    # FC와 ROI 순서가 같아야 k번째 edge가 같은 영역쌍이 된다.
    if ref_names != fc_names:
        raise RuntimeError("SC 라벨 순서가 fc_strength_68_16.csv와 다름")

    c = np.stack(mats)                           # (n_sub, 84, 84)
    iu = np.triu_indices(N_ROI, k=1)
    edges = c[:, iu[0], iu[1]]
    pair_names = [f"{ref_names[i]}__{ref_names[j]}" for i, j in zip(*iu)]
    if pair_names != fc_edge_names:
        raise RuntimeError("SC edge 순서가 fc_full_84x84_edited.csv와 다름")

    strengths = np.array([bct.strengths_und(w) for w in c])
    assert np.allclose(strengths, c.sum(axis=2)), "bct strength가 행 합과 다름"

    def save(values, colnames, path, what):
        frame = pd.DataFrame(values.astype(np.int64), columns=colnames)
        frame.insert(0, "rid", rids)
        out = clin.merge(frame, on="rid", how="inner")
        assert list(out.columns[:7]) == FINAL_COLS, f"앞 7개 컬럼이 예상과 다름: {list(out.columns[:7])}"
        out.to_csv(path, index=False)
        print(f"  {path:36s} {out.shape[0]:3d} x {out.shape[1]:<5d} (임상 7 + {what} {len(colnames)})")

    print(f"\n[{tag}] {len(rids)}명  ({root})")
    save(strengths, [f"{n}_sc" for n in ref_names],
         os.path.join(OUT_DIR, f"sc_strength_84_{tag}.csv"), "strength")
    save(edges, [f"{n}_sc" for n in pair_names],
         os.path.join(OUT_DIR, f"sc_full_84x84_{tag}.csv"), "edge")

    density = (edges > 0).mean(axis=1)
    degree = (c > 0).sum(axis=2)
    print(f"  행렬 합(사람별)   : 평균 {edges.sum(axis=1).mean():,.0f}  ({edges.sum(axis=1).min():,.0f} ~ {edges.sum(axis=1).max():,.0f})")
    print(f"  밀도              : 평균 {density.mean():.3f}  ({density.min():.3f} ~ {density.max():.3f})")
    print(f"  degree(partner 수): 평균 {degree.mean():.1f}, 최소 {degree.min()}")
    print(f"  strength          : 중앙 {np.median(strengths):,.0f}, 최소 {strengths.min():,.0f}, 최대 {strengths.max():,.0f}, 0인 칸 {(strengths == 0).sum()}")


def main():
    clin = pd.read_csv(CLIN_CSV, skipinitialspace=True)
    clin.columns = clin.columns.str.strip()
    clin = clin[FINAL_COLS]

    fc_s = pd.read_csv(FC_STRENGTH_CSV, skipinitialspace=True, nrows=0)
    fc_names = [c.strip()[:-len("_fc")] for c in fc_s.columns[7:]]
    fc_e = pd.read_csv(FC_EDGES_CSV, skipinitialspace=True, nrows=0)
    fc_edge_names = [c.strip() for c in fc_e.columns[7:]]
    assert len(fc_names) == N_ROI and len(fc_edge_names) == N_ROI * (N_ROI - 1) // 2

    os.makedirs(OUT_DIR, exist_ok=True)
    for tag, root in SC_ROOTS.items():
        build(tag, root, clin, fc_names, fc_edge_names)

    print("\n완료. 값은 원래 count다. SC를 피처로 쓸 때는 노트북에서 np.log1p를 적용한다.")


if __name__ == "__main__":
    main()
