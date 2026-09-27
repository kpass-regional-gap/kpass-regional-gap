"""전체 PCA와 인프라·차량의존·고령화 지수 생성."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import PREDICTORS, make_dirs, load_dataset, require_variation, save_csv, validate

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "output" / "tables"


def main():
    make_dirs()
    data = load_dataset(ROOT / "data" / "processed" / "analysis_dataset.csv")
    variables = PREDICTORS[:4]
    require_variation(data[variables])
    z = StandardScaler().fit_transform(data[variables])
    pca = PCA(svd_solver="full").fit(z)
    # 적재값은 변수와 주성분 점수의 상관계수로 정의한다.
    # StandardScaler의 ddof=0과 PCA 설명분산의 ddof=1 차이를 보정한다.
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_ / z.var(axis=0, ddof=1)[:, None])
    rows = []
    for pc in range(len(variables)):
        for j, variable in enumerate(variables):
            rows.append({"주성분": f"PC{pc + 1}", "변수명": variable,
                         "적재값": loadings[j, pc], "고유벡터계수": pca.components_[pc, j],
                         "설명분산비율": pca.explained_variance_ratio_[pc]})
    save_csv(pd.DataFrame(rows), TABLES / "pca_loadings_all.csv")
    for sign, label in [(1, "양의 방향"), (-1, "음의 방향")]:
        strong = [v for j, v in enumerate(variables) if sign * loadings[j, 0] >= .5]
        print(f"PC1 강한 적재(|적재값|>=0.5), {label}: {', '.join(strong) or '없음'}")
    pair = z[:, 1:3]
    pair_pca = PCA(n_components=1, svd_solver="full").fit(pair)
    weights = pair_pca.components_[0].copy()
    # 부호 임의성을 제거하여 자가용등록대수 증가 방향을 양수로 통일한다.
    orientation = 1 if weights[0] >= 0 else -1
    score = pair_pca.transform(pair)[:, 0] * orientation
    z_mean = pair.mean(axis=1)
    correlation = np.corrcoef(pair.T)[0, 1]
    ranks = pd.Series(score).rank()
    mean_ranks = pd.Series(z_mean).rank()
    same = np.array_equal(ranks, mean_ranks)
    print(f"차량 두 변수 상관 r={correlation:.6f}, PC1과 두 변수 z평균의 순위 일치: {same}")
    # 두 표준화 변수의 상관이 양수일 때만 PC1 = sqrt(2) * z평균이다.
    # 음의 상관이면 PC1은 차이 방향이며, 0이면 첫 축이 유일하지 않다.
    if correlation > 1e-10:
        assert np.allclose(score, np.sqrt(2) * z_mean), "양의 상관에서 PCA/z평균 관계 오류"
    else:
        print("주의: 상관이 양수가 아니므로 PCA와 z평균의 동일 순위는 보장되지 않습니다.")
    comparison = pd.DataFrame({"시도명": data["시도명"], "차량PC1": score,
                               "차량의존지수": z_mean, "PCA순위": ranks, "z평균순위": mean_ranks})
    save_csv(comparison, TABLES / "pca_index_comparison.csv")
    data["인프라지수"] = z[:, 0]
    data["차량의존지수"] = z_mean
    data["고령화지수"] = z[:, 3]
    validate(data)
    save_csv(data, ROOT / "output" / "processed" / "analysis_dataset_with_indices.csv")
    print("PCA 적재값, 순위 비교 및 지수 데이터 저장 완료")


if __name__ == "__main__":
    main()
