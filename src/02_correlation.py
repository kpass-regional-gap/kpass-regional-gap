"""미경험률과 설명변수·중간지표의 상관분석."""
from pathlib import Path
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from utils import PREDICTORS, make_dirs, load_dataset, require_variation, save_csv

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"


def main():
    make_dirs()
    data = load_dataset(ROOT / "data" / "processed" / "analysis_dataset.csv")
    variables = PREDICTORS + ["성인대중교통이용률"]
    require_variation(data[variables + ["미경험률"]])
    rows = []
    for variable in variables:
        r, p = pearsonr(data[variable], data["미경험률"])
        rho, sp = spearmanr(data[variable], data["미경험률"])
        rows.append({"변수명": variable, "pearson_r": r, "pearson_p": p,
                     "spearman_rho": rho, "spearman_p": sp,
                     "유의여부": bool(p < .05 and sp < .05),
                     "역할": "중간지표(종속변수아님)" if variable == "성인대중교통이용률" else "독립변수"})
    save_csv(pd.DataFrame(rows), TABLES / "correlation_results.csv")
    matrix = data[variables].corr(method="pearson")
    save_csv(matrix.rename_axis("변수명"), TABLES / "correlation_matrix.csv", index=True)
    for a, b in combinations(variables, 2):
        if abs(matrix.loc[a, b]) >= .8:
            print(f"경고: {a} / {b}: Pearson r={matrix.loc[a, b]:.4f}")
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    else:
        raise RuntimeError("한글 글꼴을 설치하세요: Malgun Gothic 또는 NanumGothic 등")
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")
    ax.scatter(data["주간평균배차간격"], data["미경험률"], color="#21618c")
    for _, row in data.iterrows():
        # 가까운 두 지역의 라벨이 겹치지 않도록 전남 라벨을 아래에 배치한다.
        offset = (5, -13) if row["시도명"] == "전라남도" else (5, 5)
        ax.annotate(row["시도명"], (row["주간평균배차간격"], row["미경험률"]),
                    xytext=offset, textcoords="offset points", fontsize=8)
    ax.set(xlabel="주간평균배차간격(분)", ylabel="K-패스 미경험률(%)",
           title="시도별 배차간격과 K-패스 미경험률")
    ax.margins(x=.2, y=.15)
    ax.grid(alpha=.2)
    fig.savefig(FIGURES / "nonexperience_vs_headway.png", dpi=180)
    plt.close(fig)
    print("상관분석 6개 변수, 상관행렬 및 산점도 저장 완료")


if __name__ == "__main__":
    main()
