"""기준 모형의 표본 및 배차간격 정의에 대한 민감도 분석."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from utils import load_dataset, make_dirs, require_variation, save_csv

ROOT = Path(__file__).resolve().parents[1]
FACTORS = ["인프라지수", "차량의존지수", "고령화지수"]


def fit_model(data):
    assert len(FACTORS) <= 4, "설명변수는 4개 이하"
    x, y = data[FACTORS], data["미경험률"]
    assert np.isfinite(data[FACTORS + ["미경험률"]].to_numpy()).all(), "결측 또는 비정상 수치"
    require_variation(data[FACTORS + ["미경험률"]])
    design = sm.add_constant(x, has_constant="add")
    assert np.linalg.matrix_rank(design) == design.shape[1], "완전 공선성"
    fit = sm.OLS(y, design, missing="raise").fit()
    # 표본 제외 시에도 지수 정의는 유지하되 beta의 표준편차는 해당 표본으로 계산한다.
    beta = fit.params[FACTORS] * x.std(ddof=0) / y.std(ddof=0)
    return fit, beta


def main():
    make_dirs()
    data = load_dataset(ROOT / "output" / "processed" / "analysis_dataset_with_indices.csv")
    baseline, _ = fit_model(data)
    # 부호가 아니라 절댓값 잔차가 가장 큰 한 지역을 전체 17개에서 제외한다.
    outlier = data.loc[baseline.resid.abs().idxmax(), "시도명"]
    weekday = data.copy()
    require_variation(weekday[["평일평균배차간격"]])
    weekday["인프라지수"] = StandardScaler().fit_transform(weekday[["평일평균배차간격"]])[:, 0]
    cases = [
        ("기준 모형", data),
        ("수도권 제외", data.loc[~data["시도명"].isin(["서울특별시", "인천광역시", "경기도"])]),
        ("서울 제외", data.loc[data["시도명"] != "서울특별시"]),
        ("평일평균배차간격", weekday),
        (f"최대절대잔차 제외({outlier})", data.loc[data["시도명"] != outlier]),
    ]
    rows = []
    for condition, sample in cases:
        fit, beta = fit_model(sample)
        row = {"조건": condition, "n": len(sample)}
        for factor in FACTORS:
            row[f"{factor}_표준화계수"] = beta[factor]
            row[f"{factor}_p"] = fit.pvalues[factor]
        row["R2"] = fit.rsquared
        rows.append(row)
        print(f"{condition}: n={len(sample)}, R²={fit.rsquared:.4f}")
    results = pd.DataFrame(rows)
    save_csv(results, ROOT / "output" / "tables" / "robustness_check.csv")
    for row in rows[1:]:
        changed = False
        for factor in FACTORS:
            sign_change = np.sign(row[f"{factor}_표준화계수"]) != np.sign(rows[0][f"{factor}_표준화계수"])
            significance_change = (row[f"{factor}_p"] < .05) != (rows[0][f"{factor}_p"] < .05)
            if sign_change or significance_change:
                changed = True
                print(f"경고: {row['조건']} / {factor}: 부호 변화={sign_change}, 유의성(p<0.05) 변화={significance_change}")
        if not changed:
            print(f"{row['조건']}: 기준 대비 부호·유의성 변화 없음")
    print("강건성 5개 조건 저장 완료")


if __name__ == "__main__":
    main()
