"""탄소 분석(08~10) 공통: 원자료 읽기, 시도명 정리, 합계 검증."""
from pathlib import Path

import numpy as np
import pandas as pd

from utils import REGIONS, normalize_regions

ROOT = Path(__file__).resolve().parents[1]
RAW_CARBON = ROOT / "data" / "raw" / "carbon"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"
PROCESSED = ROOT / "output" / "processed"
CO2_PER_C = 44 / 12          # tC -> tCO2
NATION = "전국"              # 분석용 전국 = 17개 시도 합계


def read_raw(name):
    return pd.read_csv(RAW_CARBON / name, comment="#", encoding="utf-8-sig")


def split_nation(frame, column):
    """원자료의 전국 행은 검증용으로만 분리하고 17개 시도 행을 돌려준다."""
    nation = frame[frame[column] == NATION]
    regions = frame[frame[column] != NATION].copy()
    return regions, nation


def to_wide(frame, value, columns="연료"):
    wide = frame.pivot(index="시도명", columns=columns, values=value).reset_index()
    wide.columns.name = None
    wide = normalize_regions(wide)
    return wide.set_index("시도명").loc[sorted(REGIONS, key=_region_order)]


def _region_order(name):
    order = ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시",
             "울산광역시", "세종특별자치시", "경기도", "강원특별자치도", "충청북도", "충청남도",
             "전북특별자치도", "전라남도", "경상북도", "경상남도", "제주특별자치도"]
    return order.index(name)


def check_sum(parts, total, tolerance, label):
    gap = np.abs(np.asarray(parts, dtype=float) - np.asarray(total, dtype=float))
    assert (gap <= tolerance).all(), f"{label}: 합계 불일치 최대 {gap.max()}"


def load_mileage():
    """TMACS 2024 비사업용 승용 연료별 연간 주행거리(km)와 1일 평균주행거리(km/대)."""
    raw = read_raw("tmacs_2024_nonbusiness_passenger_km_by_fuel.csv")
    assert set(raw["용도"]) == {"비사업용"} and set(raw["차종"]) == {"승용차"}
    raw = raw.rename(columns={"시도": "시도명"})
    regions, nation = split_nation(raw, "시도명")
    annual = to_wide(regions, "연간주행거리_천km") * 1000
    daily = to_wide(regions, "1일평균주행거리_km")
    fuels = ["휘발유", "경유", "LPG", "기타연료"]
    # 원자료는 천km 단위 반올림값이므로 연료 합과 계, 시도 합과 전국은 ±2천km 안에서 일치해야 한다.
    check_sum(annual[fuels].sum(axis=1), annual["계"], 2000, "TMACS 연료 합계")
    published = nation.set_index("연료")["연간주행거리_천km"] * 1000
    check_sum(annual.sum().loc[published.index], published, 2000, "TMACS 시도 합계")
    assert (annual > 0).all().all() and (daily > 0).all().all()
    return annual, daily


def load_registrations():
    """국토교통부 2024.6 승용 비사업용(관용+자가용) 연료별 등록대수."""
    raw = read_raw("molit_registration_2024_06_nonbusiness_passenger_by_fuel.csv")
    assert set(raw["차종"]) == {"승용"} and set(raw["용도"]) == {"비사업용"}
    raw = raw.rename(columns={"시도": "시도명"})
    regions, nation = split_nation(raw, "시도명")
    wide = to_wide(regions, "등록대수")
    fuels = [c for c in wide.columns if c != "총계"]
    check_sum(wide[fuels].sum(axis=1), wide["총계"], 0, "등록대수 연료 합계")
    published = nation.set_index("연료")["등록대수"]
    check_sum(wide.sum().loc[published.index], published, 0, "등록대수 시도 합계")
    return wide


def load_population():
    """행정안전부 2024.6 주민등록인구(거주자+거주불명자+재외국민)."""
    raw = read_raw("mois_population_2024_06.csv")
    regions, nation = split_nation(raw, "시도명")
    regions = normalize_regions(regions[["시도명", "총인구수"]])
    check_sum(regions["총인구수"].sum(), nation["총인구수"].iloc[0], 0, "인구 시도 합계")
    assert (regions["총인구수"] > 0).all()
    return regions.set_index("시도명")["총인구수"].loc[sorted(REGIONS, key=_region_order)]


def load_coefficients():
    raw = read_raw("emission_coefficients.csv")
    assert raw["항목"].is_unique and np.isfinite(raw["값"]).all()
    return raw.set_index("항목")["값"].to_dict()


def load_emission_factors():
    """08 산출물: 경유·LPG gCO2/km. 08을 먼저 실행해야 한다."""
    path = TABLES / "emission_factors.csv"
    assert path.exists(), "08_emission_factors.py를 먼저 실행하세요."
    table = pd.read_csv(path, encoding="utf-8-sig").set_index("구분")
    return float(table.loc["경유", "gCO2_per_km"]), float(table.loc["LPG", "gCO2_per_km"])


def ev_factor(coef, efficiency):
    """전기차 gCO2/km = 소비단 전력 CO2(tCO2/MWh = kgCO2/kWh) x 1000 / 전비(km/kWh)."""
    return coef["전력_소비단_CO2"] * 1000 / efficiency


def log_point(region_value, nation_value):
    return 100 * np.log(region_value / nation_value)
