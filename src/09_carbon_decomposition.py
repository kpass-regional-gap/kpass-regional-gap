"""보고서 그림 2: 전국 대비 1인당 승용차 주행거리와 경유·LPG CO2 격차의 로그포인트 분해."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from carbon_utils import (FIGURES, NATION, PROCESSED, TABLES, load_emission_factors, load_mileage,
                          load_population, load_registrations, log_point)
from figure_style import AXIS, DECOMP_COLORS, GRID, SHORT_NAME, use_korean_font
from utils import make_dirs, save_csv

TERMS_A = {"보유": "N/P", "주행": "V/N"}
TERMS_B = {"보유": "N/P", "주행": "V/N", "강도": "E/V"}
LABELS = {"보유": "보유 수준(대/인)", "주행": "대당 주행거리(km/대)", "강도": "1km당 배출(gCO2/km)"}


def build_dataset():
    annual, daily = load_mileage()
    cars = load_registrations()
    population = load_population()
    e_diesel, e_lpg = load_emission_factors()
    data = pd.DataFrame({
        "P_인구": population,
        "N_전체": cars["총계"],
        "V_전체_km": annual["계"],
        # 탄소는 하이브리드 분류를 등록·주행 통계 사이에서 맞추기 어려워 경유·LPG(하이브리드 제외)로 한정
        "N_경유": cars["경유"], "N_LPG": cars["엘피지"],
        "V_경유_km": annual["경유"], "V_LPG_km": annual["LPG"],
    })
    data["N_경유LPG"] = data["N_경유"] + data["N_LPG"]
    data["V_경유LPG_km"] = data["V_경유_km"] + data["V_LPG_km"]
    data["E_tCO2"] = (data["V_경유_km"] * e_diesel + data["V_LPG_km"] * e_lpg) / 1e6
    data.index.name = "시도명"
    # 일평균 주행거리는 대조용: 연간 주행거리/등록대수/365와 TMACS 1일 평균을 나란히 저장한다.
    check = pd.DataFrame({
        "TMACS_1일평균_전체": daily["계"], "연간/등록/365_전체": data["V_전체_km"] / data["N_전체"] / 365,
        "TMACS_1일평균_경유": daily["경유"], "연간/등록/365_경유": data["V_경유_km"] / data["N_경유"] / 365,
        "TMACS_1일평균_LPG": daily["LPG"], "연간/등록/365_LPG": data["V_LPG_km"] / data["N_LPG"] / 365,
    })
    return data, check


def decompose(data):
    nation = data.sum()   # 전국은 17개 시도 합계
    out = pd.DataFrame(index=data.index)
    ratios = {
        "a": {"N/P": data["N_전체"] / data["P_인구"], "V/N": data["V_전체_km"] / data["N_전체"],
              "V/P": data["V_전체_km"] / data["P_인구"]},
        "b": {"N/P": data["N_경유LPG"] / data["P_인구"], "V/N": data["V_경유LPG_km"] / data["N_경유LPG"],
              "E/V": data["E_tCO2"] * 1e6 / data["V_경유LPG_km"], "E/P": data["E_tCO2"] * 1e3 / data["P_인구"]},
    }
    national = {
        "a": {"N/P": nation["N_전체"] / nation["P_인구"], "V/N": nation["V_전체_km"] / nation["N_전체"],
              "V/P": nation["V_전체_km"] / nation["P_인구"]},
        "b": {"N/P": nation["N_경유LPG"] / nation["P_인구"], "V/N": nation["V_경유LPG_km"] / nation["N_경유LPG"],
              "E/V": nation["E_tCO2"] * 1e6 / nation["V_경유LPG_km"], "E/P": nation["E_tCO2"] * 1e3 / nation["P_인구"]},
    }
    for panel, terms, total in [("a", TERMS_A, "V/P"), ("b", TERMS_B, "E/P")]:
        for name, ratio in terms.items():
            out[f"{panel}_{name}_lp"] = log_point(ratios[panel][ratio], national[panel][ratio])
        out[f"{panel}_합계_lp"] = log_point(ratios[panel][total], national[panel][total])
        # 항등식 확인: 각 항의 로그포인트 합 = 1인당 격차의 로그포인트
        parts = out[[f"{panel}_{name}_lp" for name in terms]].sum(axis=1)
        assert np.allclose(parts, out[f"{panel}_합계_lp"]), f"패널 {panel} 분해 항등식 불일치"
        largest = out[[f"{panel}_{name}_lp" for name in terms]].abs().idxmax(axis=1)
        out[f"{panel}_절댓값최대항"] = largest.str.split("_").str[1]
    levels = pd.DataFrame({
        "1인당_차량km": ratios["a"]["V/P"], "보유_대per인_전체": ratios["a"]["N/P"],
        "대당_km_전체": ratios["a"]["V/N"], "1인당_kgCO2_경유LPG": ratios["b"]["E/P"],
        "보유_대per인_경유LPG": ratios["b"]["N/P"], "대당_km_경유LPG": ratios["b"]["V/N"],
        "gCO2_per_km_경유LPG": ratios["b"]["E/V"]})
    levels.loc[NATION] = [national["a"]["V/P"], national["a"]["N/P"], national["a"]["V/N"],
                          national["b"]["E/P"], national["b"]["N/P"], national["b"]["V/N"],
                          national["b"]["E/V"]]
    return out, levels, nation


def draw(table):
    use_korean_font()
    order = table.sort_values("b_합계_lp").index          # 아래에서 위로 그리므로 오름차순
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 8.3), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.1, wspace=0.08)
    panels = [("a", TERMS_A, "(a) 비사업용 승용 전체: 1인당 차량km"),
              ("b", TERMS_B, "(b) 경유·LPG 차량군: 1인당 CO2")]
    for ax, (panel, terms, title) in zip(axes, panels):
        right = np.zeros(len(order))
        left = np.zeros(len(order))
        for name in terms:
            values = table.loc[order, f"{panel}_{name}_lp"].to_numpy()
            start = np.where(values >= 0, right, left)
            ax.barh(y, values, left=start, height=0.62, color=DECOMP_COLORS[name],
                    edgecolor="white", linewidth=0.6, zorder=2)
            right = np.where(values >= 0, right + values, right)
            left = np.where(values < 0, left + values, left)
        total = table.loc[order, f"{panel}_합계_lp"].to_numpy()
        ax.scatter(total, y, s=46, color="black", edgecolor="white", linewidth=1.6, zorder=3)
        for index in (int(np.argmax(total)), int(np.argmin(total))):
            value = total[index]
            ax.annotate(f"{value:+.0f}", (value, y[index]), xytext=(9 if value > 0 else -9, 0),
                        textcoords="offset points", va="center", ha="left" if value > 0 else "right",
                        fontsize=12)
        ax.axvline(0, color=AXIS, linewidth=1.6, zorder=2.5)
        ax.set_xlim(-68, 42)
        ax.set_xticks(range(-60, 41, 20))
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0, labelsize=12, colors="#444444")
        ax.set_title(title, loc="left", fontsize=14)
        ax.set_xlabel("전국 대비 로그포인트(lp = 100×ln 비)", fontsize=12, color="#444444")
    axes[0].set_yticks(y, [SHORT_NAME[name] for name in order], fontsize=12.5, color="black")
    axes[0].set_ylim(-0.7, len(order) - 0.3)
    handles = [Patch(color=DECOMP_COLORS[name], label=LABELS[name]) for name in TERMS_B]
    handles.append(Line2D([], [], marker="o", linestyle="", color="black", markeredgecolor="white",
                          markersize=8, label="총격차(합)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, 0.995), handlelength=1.2, columnspacing=2.0)
    path = FIGURES / "fig2_carbon_decomposition.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    make_dirs()
    data, check = build_dataset()
    table, levels, nation = decompose(data)
    save_csv(data.reset_index(), PROCESSED / "carbon_dataset.csv")
    save_csv(table.round(6).reset_index(), TABLES / "carbon_decomposition.csv")
    save_csv(levels.rename_axis("시도명").reset_index(), TABLES / "carbon_levels.csv")
    save_csv(check.round(3).reset_index(), TABLES / "carbon_daily_km_check.csv")

    share_v = nation["V_경유LPG_km"] / nation["V_전체_km"]
    share_n = nation["N_경유LPG"] / nation["N_전체"]
    per_capita = levels["1인당_kgCO2_경유LPG"]
    own_a = int((table["a_절댓값최대항"] == "보유").sum())
    own_b = int((table["b_절댓값최대항"] == "보유").sum())
    summary = pd.DataFrame([
        ("경유·LPG 차량km 비중(%)", share_v * 100), ("경유·LPG 등록대수 비중(%)", share_n * 100),
        ("경유·LPG 기준배출(만 tCO2/년)", nation["E_tCO2"] / 1e4),
        ("1인당 배출 전국(kgCO2/년)", per_capita[NATION]),
        ("1인당 배출 서울(kgCO2/년)", per_capita["서울특별시"]),
        ("1인당 배출 전남(kgCO2/년)", per_capita["전라남도"]),
        ("보유 항 절댓값 최대 시도 수(a)", own_a), ("보유 항 절댓값 최대 시도 수(b)", own_b),
    ], columns=["항목", "값"])
    save_csv(summary, TABLES / "carbon_summary.csv")
    print(summary.round(2).to_string(index=False))
    for name in ["서울특별시", "전라남도"]:
        row = table.loc[name]
        print(f"{name}: (a) 보유 {row['a_보유_lp']:.1f}, 주행 {row['a_주행_lp']:.1f} / "
              f"(b) 보유 {row['b_보유_lp']:.1f}, 주행 {row['b_주행_lp']:.1f}, 강도 {row['b_강도_lp']:.1f}, 합계 {row['b_합계_lp']:.1f}")
    print("보유 항이 최대가 아닌 시도(b):", ", ".join(table.index[table["b_절댓값최대항"] != "보유"]))
    path = draw(table)
    print(f"분해표·요약표 및 그림 2 저장 완료: {path.name}")


if __name__ == "__main__":
    main()
