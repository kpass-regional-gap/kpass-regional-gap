"""보고서 그림 3: 경유·LPG 차량군의 거리 감축(A)·전기차 전환(B)·병행(C) 조건부 CO2 감축량."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from carbon_utils import FIGURES, NATION, PROCESSED, TABLES, ev_factor, load_coefficients, load_emission_factors
from figure_style import DECOMP_COLORS, GRID, use_korean_font
from utils import make_dirs, normalize_regions, save_csv

LABELS = {"A": "A 주행거리 감축\n(a = 1%)", "B": "B 전기차 전환\n(b = 1%)", "C": "C 병행\n(a = b = 1%)"}


def scenarios(v_diesel, v_lpg, e_diesel, e_lpg, e_ev, a, b):
    """차량km(km)와 gCO2/km로 tCO2 감축량 계산. 비율은 차량 대수가 아닌 차량km 기준이다."""
    base = (v_diesel * e_diesel + v_lpg * e_lpg) / 1e6
    reduce_a = a * base                                                     # A = Σ a V e
    reduce_b = b * (v_diesel * (e_diesel - e_ev) + v_lpg * (e_lpg - e_ev)) / 1e6   # B = Σ b V (e - e_EV)
    # C: 먼저 a만큼 줄이고 남은 (1-a)V에만 전환을 적용해 같은 차량km를 두 번 감축하지 않는다.
    reduce_c = (a * (v_diesel * e_diesel + v_lpg * e_lpg)
                + b * (1 - a) * (v_diesel * (e_diesel - e_ev) + v_lpg * (e_lpg - e_ev))) / 1e6
    assert np.allclose(reduce_c, reduce_a + (1 - a) * reduce_b), "C 중복 감축 점검 실패"
    return base, {"A": reduce_a, "B": reduce_b, "C": reduce_c}


def main():
    make_dirs()
    data = normalize_regions(pd.read_csv(PROCESSED / "carbon_dataset.csv", encoding="utf-8-sig"))
    data = data.set_index("시도명")
    coef = load_coefficients()
    e_diesel, e_lpg = load_emission_factors()
    a, b = coef["시나리오_a_거리감축비율"], coef["시나리오_b_전기차전환비율"]
    efficiencies = {"하한": coef["전기차_전비_하한"], "기준": coef["전기차_전비_기준"], "상한": coef["전기차_전비_상한"]}

    rows, regional = [], []
    totals = data[["V_경유_km", "V_LPG_km"]].sum()     # 전국 = 17개 시도 합계
    for label, efficiency in efficiencies.items():
        e_ev = ev_factor(coef, efficiency)
        base, result = scenarios(totals["V_경유_km"], totals["V_LPG_km"], e_diesel, e_lpg, e_ev, a, b)
        for name, value in result.items():
            rows.append({"시나리오": name, "전비구분": label, "전비_km_per_kWh": efficiency,
                         "전기차_gCO2_per_km": e_ev, "기준배출_tCO2": base, "감축량_tCO2": value,
                         "감축량_만tCO2": value / 1e4, "기준대비_%": value / base * 100})
        if label == "기준":
            region_base, region_result = scenarios(data["V_경유_km"], data["V_LPG_km"], e_diesel, e_lpg, e_ev, a, b)
            regional = pd.DataFrame({"기준배출_tCO2": region_base,
                                     **{f"{k}_감축_tCO2": v for k, v in region_result.items()}})
            regional.loc[NATION] = regional.sum()
    national = pd.DataFrame(rows)
    save_csv(national, TABLES / "carbon_scenarios.csv")
    save_csv(regional.rename_axis("시도명").reset_index(), TABLES / "carbon_scenarios_by_region.csv")

    base_case = national[national["전비구분"] == "기준"].set_index("시나리오")
    span = national.pivot(index="시나리오", columns="전비구분", values="감축량_만tCO2")
    print(f"기준 배출: {base_case['기준배출_tCO2'].iloc[0] / 1e4:,.1f}만 tCO2/년")
    for name in LABELS:
        row = base_case.loc[name]
        print(f"{name}: {row['감축량_만tCO2']:.1f}만 t ({row['기준대비_%']:.2f}%), "
              f"전비 4.5~5.5: {span.loc[name, '하한']:.1f}~{span.loc[name, '상한']:.1f}만 t")

    use_korean_font()
    fig, ax = plt.subplots(figsize=(13.2, 3.9))
    fig.subplots_adjust(left=0.14, right=0.98, top=0.97, bottom=0.2)
    names = list(LABELS)
    y = np.arange(len(names))[::-1]
    values = base_case.loc[names, "감축량_만tCO2"].to_numpy()
    ax.barh(y, values, height=0.46, color=DECOMP_COLORS["보유"], zorder=2)
    for yi, name, value in zip(y, names, values):
        low, high = sorted([span.loc[name, "하한"], span.loc[name, "상한"]])
        end = value
        if high - low > 1e-9:   # A는 전비와 무관하므로 범위가 없다. 검은 선은 신뢰구간이 아니라 전비 설정 범위
            ax.errorbar(value, yi, xerr=[[value - low], [high - value]], fmt="none", ecolor="black",
                        elinewidth=1.6, capsize=6, capthick=1.6, zorder=3)
            end = high
        ax.text(end + 0.7, yi, f"{value:.1f}만 tCO2 ({base_case.loc[name, '기준대비_%']:.2f}%)",
                va="center", fontsize=13)
    ax.set_yticks(y, [LABELS[n] for n in names], fontsize=13)
    ax.set_xlim(0, np.ceil(values.max() / 10) * 10 + 8)
    ax.set_ylim(-0.55, len(names) - 0.45)
    ax.set_xlabel("연간 CO2 감소량(만 tCO2/년, 17개 시도 합계)", fontsize=12, color="#444444")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=12, colors="#444444")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    path = FIGURES / "fig3_carbon_scenarios.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"시나리오 표 및 그림 3 저장 완료: {path.name}")


if __name__ == "__main__":
    main()
