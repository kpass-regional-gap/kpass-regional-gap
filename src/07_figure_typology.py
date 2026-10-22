"""보고서 그림 1: 06 유형표의 종합지수 z를 지역별 막대사탕 그림으로 재배치."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import (AXIS, BLUE, BLUE_LIGHT, ORANGE, ORANGE_LIGHT, SHORT_NAME,
                          clean_axes, use_korean_font)
from utils import COMPOSITE, make_dirs, normalize_regions

ROOT = Path(__file__).resolve().parents[1]
GOOD, WEAK = "양호형", "인프라·차량의존 취약형"


def main():
    make_dirs()
    table = normalize_regions(pd.read_csv(ROOT / "output" / "tables" / "region_typology.csv",
                                          encoding="utf-8-sig"))
    z = table[f"{COMPOSITE}_z"]
    assert np.isfinite(z).all(), "06단계를 먼저 실행하세요."
    assert table["유형"].isin([GOOD, WEAK]).all(), "기준 모형에서 종합지수가 유의할 때만 그림 1을 그립니다."
    # 06의 분류 규칙(z<=0 양호형)과 저장된 유형이 같은지 다시 확인한다.
    assert ((z <= 0) == (table["유형"] == GOOD)).all(), "유형과 z 부호가 일치하지 않습니다."

    use_korean_font()
    fig, ax = plt.subplots(figsize=(9.2, 5.3), layout="constrained")
    # 06 산점도와 같은 지역 순서(위에서부터 제주 ... 서울). 세로축은 지역명일 뿐 분석변수가 아니다.
    y = np.arange(len(table))
    below = table["유형"] == GOOD
    for i, (value, low) in enumerate(zip(z, below)):
        ax.plot([0, value], [y[i], y[i]], color=BLUE_LIGHT if low else ORANGE_LIGHT,
                linewidth=2.6, solid_capstyle="butt", zorder=2)
        offset = -0.07 if value < 0 else 0.07
        ax.text(value + offset, y[i], f"{value:.2f}", va="center",
                ha="right" if value < 0 else "left", fontsize=11)
    ax.scatter(z[below], y[below], s=70, color=BLUE, zorder=3,
               label=f"평균 이하 {int(below.sum())}곳")
    ax.scatter(z[~below], y[~below], s=70, color=ORANGE, zorder=3,
               label=f"평균 초과 {int((~below).sum())}곳")
    ax.axvline(0, color="#888888", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_yticks(y, [SHORT_NAME[name] for name in table["시도명"]], fontsize=13)
    ax.set_xlim(np.floor(z.min()) - 0.1, z.max() + 0.5)
    ax.set_xticks(np.arange(np.floor(z.min()), np.floor(z.max()) + 1))
    ax.set_ylim(-0.7, len(table) - 0.4)
    ax.set_xlabel("인프라·차량의존 종합지수(z)", fontsize=13)
    clean_axes(ax)
    ax.spines["bottom"].set_color(AXIS)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              fontsize=12, handletextpad=0.4, columnspacing=2.5)
    path = ROOT / "output" / "figures" / "fig1_typology_lollipop.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"평균 이하 {int(below.sum())}곳: {', '.join(SHORT_NAME[n] for n in table.loc[below, '시도명'])}")
    print(f"평균 초과 {int((~below).sum())}곳: {', '.join(SHORT_NAME[n] for n in table.loc[~below, '시도명'])}")
    print(f"그림 1 저장 완료: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
