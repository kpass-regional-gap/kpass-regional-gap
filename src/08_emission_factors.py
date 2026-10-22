"""2023 에너지총조사와 연료 계수로 경유·LPG 승용차의 차량km당 CO2와 전기차 배출계수를 산정."""
import numpy as np
import pandas as pd

from carbon_utils import CO2_PER_C, TABLES, ev_factor, load_coefficients, read_raw
from utils import make_dirs, save_csv

# 보고서 본문 값(소수 첫째 자리). 재현 여부만 확인하고 계산에는 쓰지 않는다.
REPORT = {"경유": 297.3, "LPG": 233.6, "전기차(5.0km/kWh)": 83.0}


def main():
    make_dirs()
    census = read_raw("energy_census_2023_passenger.csv")
    coef = load_coefficients()
    detail = census[census["구분"] == "세부"].copy()
    subtotal = census[census["구분"] == "소계"].set_index(["연료", "차형"])
    # 옮겨 적은 값 검증: 배기량별 합계가 원자료 소계와 반올림(0.1) 범위에서 일치해야 한다.
    for (fuel, body), group in detail.groupby(["연료", "차형"]):
        for column in ["연료소비량_kl", "열량_Tcal"]:
            gap = abs(group[column].sum() - subtotal.loc[(fuel, body), column])
            assert gap <= 0.2 + 1e-9, f"{fuel}/{body}/{column} 소계 불일치: {gap}"
    ratio = detail["대당연간주행거리_km"] / detail["대당연간소비량_l"]
    assert (abs(ratio - detail["연비_km_per_l"]) <= 0.05 + 1e-9).all(), "표의 연비와 주행거리/소비량 불일치"

    # 차형·배기량별 차량km = 연료소비량 / 대당 소비량 x 대당 주행거리.
    # 표의 연비는 소수 한 자리로 반올림되어 있어 대당 주행거리/대당 소비량으로 연비를 다시 구한다.
    detail["연료소비량_l"] = detail["연료소비량_kl"] * 1000
    detail["추정차량수"] = detail["연료소비량_l"] / detail["대당연간소비량_l"]
    detail["차량km"] = detail["추정차량수"] * detail["대당연간주행거리_km"]

    # 경유: 순발열량(MJ/l) x 탄소배출계수(tC/TJ = gC/MJ) x 44/12 = gCO2/l
    diesel_per_l = coef["경유_순발열량"] * coef["경유_탄소배출계수"] * CO2_PER_C
    # LPG: 차량용 LPG를 부탄으로 가정. 총조사 표 2-1-1의 열량(Tcal)/부피(kl)를 부탄 총발열량으로
    # 나누어 kg/l 밀도로 환산한 뒤 부탄 순발열량과 탄소배출계수를 적용한다.
    lpg = detail[detail["연료"] == "LPG"]
    kcal_per_l = lpg["열량_Tcal"].sum() * 1e9 / lpg["연료소비량_l"].sum()
    density = kcal_per_l / coef["부탄_총발열량"]
    lpg_per_kg = coef["부탄_순발열량"] * coef["부탄_탄소배출계수"] * CO2_PER_C
    lpg_per_l = density * lpg_per_kg
    detail["gCO2_per_l"] = np.where(detail["연료"] == "경유", diesel_per_l, lpg_per_l)
    detail["CO2_t"] = detail["연료소비량_l"] * detail["gCO2_per_l"] / 1e6
    detail["gCO2_per_km"] = detail["gCO2_per_l"] / (detail["차량km"] / detail["연료소비량_l"])
    save_csv(detail, TABLES / "emission_factor_details.csv")

    rows = []
    for fuel, per_l in [("경유", diesel_per_l), ("LPG", lpg_per_l)]:
        part = detail[detail["연료"] == fuel]
        economy = part["차량km"].sum() / part["연료소비량_l"].sum()   # 차량km 가중 평균 연비
        rows.append({"구분": fuel, "연비": economy, "연비단위": "km/l", "단위당CO2": per_l,
                     "단위당CO2단위": "gCO2/l", "gCO2_per_km": per_l / economy,
                     "비고": "부탄 계수, 밀도 %.4f kg/l(총조사 열량/부피 ÷ 부탄 총발열량)" % density
                     if fuel == "LPG" else "경유 순발열량·탄소배출계수"})
    for efficiency in sorted({coef["전기차_전비_하한"], coef["전기차_전비_기준"], coef["전기차_전비_상한"]}):
        rows.append({"구분": f"전기차({efficiency:.1f}km/kWh)", "연비": efficiency, "연비단위": "km/kWh",
                     "단위당CO2": coef["전력_소비단_CO2"] * 1000, "단위당CO2단위": "gCO2/kWh",
                     "gCO2_per_km": ev_factor(coef, efficiency), "비고": "GIR 2023 소비단 전력 CO2, 전비는 가정"})
    result = pd.DataFrame(rows)
    save_csv(result, TABLES / "emission_factors.csv")

    print(f"경유 gCO2/l={diesel_per_l:.2f}, LPG 밀도={density:.5f} kg/l, LPG gCO2/l={lpg_per_l:.2f}")
    print(result[["구분", "연비", "연비단위", "gCO2_per_km"]].round(4).to_string(index=False))
    for name, value in REPORT.items():
        computed = result.set_index("구분").loc[name, "gCO2_per_km"]
        same = round(computed, 1) == value
        print(f"{name}: 계산 {computed:.3f} g/km, 보고서 {value} g/km -> {'일치' if same else '경고: 불일치'}")
    print("배출계수 저장 완료")


if __name__ == "__main__":
    main()
