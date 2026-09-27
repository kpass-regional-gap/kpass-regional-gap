"""기준 모형의 유의 요인으로 탐색적 지역 유형 판정."""
from pathlib import Path
from importlib import import_module

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import load_dataset, make_dirs, save_csv

ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS = import_module("05_robustness")
FACTORS = ROBUSTNESS.FACTORS


def classify(values):
    if len(values) == 0:
        return "판정유보(유의요인없음)"
    ordered = values.sort_values(ascending=False)
    if ordered.iloc[0] <= 0:
        return "양호형"
    # 양수인 요인이 하나뿐이면 그 요인의 유형. 둘 이상일 때 상위 두 값을 비교한다.
    positive = ordered[ordered > 0]
    if len(positive) == 1 or positive.iloc[0] - positive.iloc[1] >= .5:
        return {"인프라지수": "인프라취약형", "차량의존지수": "차량의존형",
                "구조요인z평균": "구조취약형", "고령화지수": "고령화형"}[positive.index[0]]
    return "복합취약형"


def main():
    make_dirs()
    data = load_dataset(ROOT / "output" / "processed" / "analysis_dataset_with_indices.csv")
    # 저장된 05 결과와 현재 데이터의 기준 모형을 대조하여 오래된 결과 사용을 차단한다.
    table = pd.read_csv(ROOT / "output" / "tables" / "robustness_check.csv", encoding="utf-8-sig")
    baseline = table.loc[table["조건"] == "기준 모형"]
    assert len(baseline) == 1, "05단계를 먼저 실행하세요."
    baseline = baseline.iloc[0]
    fit, beta = ROBUSTNESS.fit_model(data)
    assert baseline["n"] == len(data) and np.isclose(baseline["R2"], fit.rsquared), "05 결과가 오래되었습니다."
    for factor in FACTORS:
        assert np.isclose(baseline[f"{factor}_p"], fit.pvalues[factor]), "05단계를 다시 실행하세요."
        assert np.isclose(baseline[f"{factor}_표준화계수"], beta[factor]), "05단계를 다시 실행하세요."
    significant = [f for f in FACTORS if baseline[f"{f}_p"] < .05]
    print(f"기준 모형 유의 요인(p<0.05): {', '.join(significant) or '없음'}")
    # 차량 지수는 두 z의 평균이므로 분산이 1이 아니다. 0.5 차이 비교를 위해
    # 유형화에서만 세 지수를 다시 표준화한다. 03의 저장 지수 정의는 유지한다.
    z = pd.DataFrame(StandardScaler().fit_transform(data[FACTORS]), columns=FACTORS)
    axes = z[significant].copy()
    influence = beta.abs().to_dict()
    if {"인프라지수", "차량의존지수"}.issubset(significant):
        axes = axes.drop(columns=["인프라지수", "차량의존지수"])
        axes["구조요인z평균"] = z[["인프라지수", "차량의존지수"]].mean(axis=1)
        influence["구조요인z평균"] = beta[["인프라지수", "차량의존지수"]].abs().mean()
    ordered_axes = sorted(axes.columns, key=lambda f: influence[f], reverse=True)
    result = data[["시도명"]].copy()
    result["유형"] = [classify(row) for _, row in axes.iterrows()]
    for factor in FACTORS:
        result[f"{factor}_z"] = z[factor]
        result[f"{factor}_유의"] = factor in significant
    if "구조요인z평균" in axes:
        result["구조요인z평균"] = axes["구조요인z평균"]
    result["판정축"] = ", ".join(ordered_axes) or "없음"
    assert len(result) == 17 and not result.isna().any().any()
    save_csv(result, ROOT / "output" / "tables" / "region_typology.csv")
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    else:
        raise RuntimeError("한글 글꼴이 필요합니다.")
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
    x = axes[ordered_axes[0]] if ordered_axes else pd.Series(np.zeros(len(data)))
    # 단일 축에서는 겹침 방지용 행 위치만 사용하며 두 번째 요인으로 해석하지 않는다.
    y = axes[ordered_axes[1]] if len(ordered_axes) > 1 else pd.Series(np.arange(len(data)) - 8)
    for kind in result["유형"].unique():
        mask = result["유형"] == kind
        ax.scatter(x[mask], y[mask], label=kind, s=45)
    for i, name in enumerate(data["시도명"]):
        offset = (5, 3) if len(ordered_axes) < 2 else (5, 6 + (i % 3) * 12)
        ax.annotate(name, (x.iloc[i], y.iloc[i]), xytext=offset,
                    textcoords="offset points", fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": "gray", "lw": .5})
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set(xlabel=ordered_axes[0] if ordered_axes else "유의 요인 없음",
           ylabel=ordered_axes[1] if len(ordered_axes) > 1 else "지역 표시 위치 (두 번째 유의 요인 없음)",
           title="유의 요인에 따른 시도 유형 (탐색적 분류)")
    ax.margins(.25)
    if len(ordered_axes) < 2:
        ax.set_yticks([])
    ax.legend()
    fig.savefig(ROOT / "output" / "figures" / "typology_scatter.png", dpi=180)
    plt.close(fig)
    print(result[["시도명", "유형"]].to_string(index=False))
    print("지역 유형표 및 산점도 저장 완료")


if __name__ == "__main__":
    main()
