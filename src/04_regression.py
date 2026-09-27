"""17개 지역의 미경험률에 대한 3개 OLS 모형."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils import make_dirs, load_dataset, require_variation

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "output" / "tables"


def main():
    make_dirs()
    data = load_dataset(ROOT / "output" / "processed" / "analysis_dataset_with_indices.csv")
    extra = ["성인대중교통이용률", "월평균대중교통비"]
    require_variation(data[extra + ["미경험률"]])
    data[[f"{v}(z)" for v in extra]] = StandardScaler().fit_transform(data[extra])
    base = ["인프라지수", "차량의존지수", "고령화지수"]
    models = [base, base + ["성인대중교통이용률(z)"], base + ["월평균대중교통비(z)"]]
    y = data["미경험률"]
    for number, variables in enumerate(models, 1):
        assert len(variables) <= 4, "절편 제외 설명변수는 4개 이하로 제한합니다."
        if len(variables) == 4:
            print(f"경고: 모형{number}: 변수 4개로 제한선에 도달, 해석에 더 유의할 것")
        x = data[variables]
        require_variation(x)
        design = sm.add_constant(x, has_constant="add")
        assert np.linalg.matrix_rank(design) == design.shape[1], "설계행렬이 완전 공선성을 가집니다."
        fit = sm.OLS(y, design, missing="raise").fit()
        # 종속·독립변수 모두 표준화했을 때의 계수와 같은 beta이다.
        beta = fit.params[variables] * x.std(ddof=0) / y.std(ddof=0)
        vif = pd.Series({v: variance_inflation_factor(design.to_numpy(), j + 1)
                         for j, v in enumerate(variables)}, name="VIF")
        report = (f"모형{number}: 미경험률 ~ {' + '.join(variables)}\n\n"
                  + fit.summary().as_text() + "\n\n표준화계수 (X와 Y 모두 표준화, 절편 제외)\n"
                  + beta.rename("표준화계수").to_string() + "\n\nVIF (절편을 포함한 설계행렬로 계산)\n"
                  + vif.to_string() + "\n")
        (TABLES / f"regression_모형{number}.txt").write_text(report, encoding="utf-8")
        residuals = pd.DataFrame({"시도명": data["시도명"], "잔차": fit.resid})
        largest = residuals.loc[residuals["잔차"].abs().nlargest(3).index]
        print(f"모형{number}: n={int(fit.nobs)}, R²={fit.rsquared:.4f}, 절댓값 잔차 상위 3개 지역")
        print(largest.to_string(index=False))
    print("OLS 모형 3개 보고서 저장 완료")


if __name__ == "__main__":
    main()
