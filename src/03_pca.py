"""전체 요인 검증용 PCA와 세 변수 PCA 종합지수 생성."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import (PREDICTORS, COMPOSITE, build_composite, make_dirs,
                   load_dataset, require_variation, save_csv, validate)

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
    score, details = build_composite(data)
    save_csv(details, TABLES / "pca_loadings_composite.csv")
    print(details.to_string(index=False))
    z_mean = z[:, :3].mean(axis=1)
    ranks = pd.Series(score).rank()
    mean_ranks = pd.Series(z_mean).rank()
    same = np.array_equal(ranks, mean_ranks)
    # 세 변수에서는 PCA와 동일 가중 z평균의 동등성을 가정하지 않는다.
    print(f"참고: 세 변수 PC1과 z평균의 순위 일치: {same} (동일성 보장 없음)")
    comparison = pd.DataFrame({"시도명": data["시도명"], COMPOSITE: score,
                               "세변수z평균": z_mean, "PCA순위": ranks, "z평균순위": mean_ranks})
    save_csv(comparison, TABLES / "pca_index_comparison.csv")
    data[COMPOSITE] = score
    # 고령화지수는 참고용으로 보관하며 회귀·유형화에서는 제외한다.
    data["고령화지수"] = z[:, 3]
    validate(data)
    save_csv(data, ROOT / "output" / "processed" / "analysis_dataset_with_indices.csv")
    print("PCA 적재값, 순위 비교 및 지수 데이터 저장 완료")


if __name__ == "__main__":
    main()
