# -*- coding: utf-8 -*-
"""
FC(84x84 DKT Combined, ses-t0)를 구조 morphometry CSV와 동일한 레이아웃으로 병합한다.
두 해상도를 모두 만들어 두고 노트북에서 골라 쓴다.

출력 1: fc_strength_68_16.csv     (44 x 91)   ROI별 node strength 84개
출력 2: fc_full_84x84_edited.csv  (44 x 3493) 상삼각 edge 3486개 전부

둘 다 앞 7개 컬럼이 rid, treatment_group, vas_t0, vas_t1, age, sex, mean_fd라
구조 CSV(structural_full_68_68_16_edited.csv)와 똑같이 index 7부터 뇌 피처가 시작한다.
따라서 ElasticNet 노트북은 파일명만 바꾸면 그대로 돌아간다.

구조 CSV의 7번째 컬럼은 eTIV(두개내용적)지만 FC에서는 그 자리에 mean FD를 넣는다.
eTIV는 피질 두께/부피를 보정할 때나 의미가 있고 상관 기반 FC와는 무관하다.
반면 head motion은 FC를 체계적으로 왜곡하는 대표적 교란변수라 반드시 빼야 한다.
mean FD = fmriprep confounds의 framewise_displacement 평균 (첫 볼륨은 NaN이라 제외).

node strength 정의
  strength_i = mean_{j != i} arctanh(r_ij)
  84x84 Combined 행렬 전체에서 계산하므로 cortical, subcortical, 그리고
  cortico-subcortical 1088개 edge가 모두 평균에 들어간다. 기존의 분리된
  Cortical(68x68) / Subcortical(16x16) 파일은 이 행렬의 부분집합이라
  따로 넣을 필요가 없다 (functional_connectivity_combined.py --validate-only로 확인됨).

strength는 행 평균이라 "이 ROI가 전반적으로 더/덜 연결돼 있나"만 답한다.
"특정 edge가 갈망과 관련 있나"를 보려면 edge 버전을 써야 한다. 다만 n=44에
피처 3486개라 p/n이 79이므로 결과 해석에 주의가 필요하다.

Pearson r은 |r|이 1에 가까울수록 분산이 커지므로 Fisher z로 변환한 뒤 쓴다.

주의: 구조 CSV 45명 중 sub-001은 ses-t0 fmriprep이 boldref_fmap에서 크래시해
MNI 공간 preproc BOLD가 없다. FC 계산이 불가능하므로 n=44가 된다.
구조 결과와 직접 비교할 때는 구조 쪽도 rid=1을 빼고 다시 돌려야 표본이 맞는다.
"""

import glob
import os
import numpy as np
import pandas as pd

FC_ROOT = r"D:\SUDMEX\Connectivity_fMRI_combined"
FMRIPREP_DIR = r"D:\SUDMEX\fmriprep_output"
CLIN_CSV = "structural_full_68_68_16_edited.csv"
OUT_STRENGTH = "fc_strength_68_16.csv"
OUT_EDGES = "fc_full_84x84_edited.csv"
SESSION = "ses-t0"

# 구조 CSV에서 가져올 임상 컬럼. eTIV는 FC와 무관하므로 빼고 mean_fd를 뒤에 붙인다.
CLIN_COLS = ["rid", "treatment_group", "vas_t0", "vas_t1", "age", "sex"]
FINAL_COLS = CLIN_COLS + ["mean_fd"]

# 구조 CSV의 피질하 명명 규칙에 맞춘다 (lh_Accumbens_vol -> lh_Accumbens_fc)
SUBCORT_RENAME = {"Accumbens-area": "Accumbens"}

# 표준형 이름에서 피질/피질하를 가르는 기준. aseg 16개 구조의 region 부분.
SUBCORT = {"Thalamus", "Caudate", "Putamen", "Pallidum",
           "Hippocampus", "Amygdala", "Accumbens", "VentralDC"}


def roi_name(label):
    """ROI 라벨 -> 구조 CSV 스타일 이름 {hemi}_{region} (측정치 접미사 없음).

    FreeSurferColorLUT 원본은 피질을 ctx-lh-bankssts, 피질하를 Left-Thalamus로
    적어 두 규칙이 섞인다. 구조 CSV(lh_bankssts_thick / lh_Thalamus_vol)가 이미
    {hemi}_{region} 하나로 흡수해 놨으므로 FC도 같은 규칙을 쓴다.
    이름이 정확히 일치해야 _thick <-> _fc 문자열만 바꿔서 두 모달리티의
    피처 표를 조인할 수 있다.

    D 드라이브 FC 파일은 standardize_fc_labels.py로 이미 표준형이지만,
    원본 형식으로 재생성된 파일도 받을 수 있게 둘 다 처리한다.
    """
    label = label.strip()
    if label.startswith(("lh_", "rh_")):
        return label                                  # 이미 표준형
    if label.startswith("ctx-"):
        _, hemi, region = label.split("-", 2)
        return f"{hemi}_{region}"
    side, region = label.split("-", 1)
    hemi = {"Left": "lh", "Right": "rh"}[side]
    return f"{hemi}_{SUBCORT_RENAME.get(region, region)}"


def sort_key(label):
    """구조 CSV 순서에 맞춘다: cortical(lh->rh) 먼저, 그 다음 subcortical(lh->rh)."""
    hemi, region = roi_name(label).split("_", 1)
    return (region in SUBCORT, hemi == "rh")


def fc_csv(rid):
    sub = f"sub-{rid:03d}"
    return os.path.join(FC_ROOT, sub, SESSION,
                        f"{sub}_{SESSION}_FreeSurferDKT_Combined.connectivity.csv")


def mean_fd(rid):
    """fmriprep confounds에서 mean framewise displacement.

    첫 볼륨의 FD는 정의상 NaN이라 nanmean으로 뺀다. FC 계산에 쓴 것과
    같은 세션(ses-t0)의 confounds 파일을 쓴다.
    """
    hits = glob.glob(os.path.join(FMRIPREP_DIR, f"sub-{rid:03d}", SESSION, "func",
                                  "*desc-confounds_timeseries.tsv"))
    if not hits:
        raise FileNotFoundError(f"sub-{rid:03d}: confounds 파일 없음")
    fd = pd.read_csv(hits[0], sep="\t")["framewise_displacement"].values
    return float(np.nanmean(fd))


def main():
    clin = pd.read_csv(CLIN_CSV, skipinitialspace=True)
    clin.columns = clin.columns.str.strip()
    clin = clin[CLIN_COLS]

    zmats, rids, fds, ref_labels = [], [], [], None
    for rid in clin["rid"]:
        p = fc_csv(rid)
        if not os.path.exists(p):
            print(f"  sub-{rid:03d}: FC({SESSION}) 없음, 제외")
            continue

        mat = pd.read_csv(p, index_col=0)
        labels = list(mat.columns)
        if ref_labels is None:
            ref_labels = labels
        elif labels != ref_labels:
            raise RuntimeError(f"sub-{rid:03d}: 라벨 순서/구성이 기준과 다름")

        r = mat.values
        if not np.allclose(r, r.T, atol=1e-8):
            raise RuntimeError(f"sub-{rid:03d}: 연결성 행렬이 대칭이 아님")

        zmats.append(np.arctanh(np.clip(r, -0.999999, 0.999999)))
        fds.append(mean_fd(rid))
        rids.append(rid)

    # 두 출력 모두 구조 CSV 순서(피질 lh->rh, 그 다음 피질하 lh->rh)로 재정렬한다.
    # 모델은 컬럼 순서를 타지 않지만 두 파일을 나란히 읽을 때 헷갈리지 않게 맞춘다.
    order = sorted(range(len(ref_labels)), key=lambda i: sort_key(ref_labels[i]))
    names = [roi_name(ref_labels[i]) for i in order]
    if len(set(names)) != len(names):
        raise RuntimeError("ROI 이름 중복 발생")

    z = np.stack(zmats)[:, order][:, :, order]      # (n_sub, 84, 84), 재정렬 완료
    if not np.all(np.isfinite(z)):
        raise RuntimeError("비유한 값 존재")

    n_sub, n_roi = z.shape[0], z.shape[1]

    # strength: 대각을 0으로 두고 나머지 83개 파트너의 평균
    zs = z.copy()
    zs[:, np.arange(n_roi), np.arange(n_roi)] = 0.0
    strengths = zs.sum(axis=2) / (n_roi - 1)
    strength_names = [f"{n}_fc" for n in names]

    # edge: 상삼각(대각 제외)만 취해 중복 제거
    iu = np.triu_indices(n_roi, k=1)
    edges = z[:, iu[0], iu[1]]
    edge_names = [f"{names[i]}__{names[j]}" for i, j in zip(*iu)]

    motion = pd.DataFrame({"rid": rids, "mean_fd": fds})

    def save(values, colnames, path, what):
        frame = pd.DataFrame(values, columns=colnames)
        frame.insert(0, "rid", rids)
        out = clin.merge(motion, on="rid").merge(frame, on="rid", how="inner")
        assert list(out.columns[:7]) == FINAL_COLS, f"앞 7개 컬럼이 예상과 다름: {list(out.columns[:7])}"
        out.to_csv(path, index=False)
        print(f"  {path:28s} {out.shape[0]:3d} x {out.shape[1]:<5d} "
              f"(임상 7 + {what} {len(colnames)})")

    n_sub_roi = sum(1 for n in names if n.split("_", 1)[1] in SUBCORT)
    print(f"\n저장 ({n_sub}명, ROI {len(names)-n_sub_roi} cortical + {n_sub_roi} subcortical):")
    save(strengths, strength_names, OUT_STRENGTH, "strength")
    save(edges, edge_names, OUT_EDGES, "edge")
    print(f"\n  strength 첫/끝 : {strength_names[0]} / {strength_names[-1]}")
    print(f"  edge 첫/끝     : {edge_names[0]} / {edge_names[-1]}")


if __name__ == "__main__":
    main()
