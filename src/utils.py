"""경로, 시도명 표준화 및 공통 검증."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"
INDICES = ROOT / "output" / "processed"
REGION_MAP = {
    "서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시",
    "인천": "인천광역시", "광주": "광주광역시", "대전": "대전광역시",
    "울산": "울산광역시", "세종": "세종특별자치시", "경기": "경기도",
    "강원": "강원특별자치도", "충북": "충청북도", "충남": "충청남도",
    "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도",
    "경남": "경상남도", "제주": "제주특별자치도",
}
REGIONS = set(REGION_MAP.values())
REGION_MAP.update({name: name for name in REGIONS})
REGION_MAP.update({"강원도": "강원특별자치도", "전라북도": "전북특별자치도",
                   "제주도": "제주특별자치도", "세종시": "세종특별자치시"})
PREDICTORS = ["주간평균배차간격", "1인당자가용등록대수", "1일평균주행거리",
              "고령인구비율", "월평균대중교통비"]


def make_dirs():
    for path in (DATA, TABLES, FIGURES, INDICES):
        path.mkdir(parents=True, exist_ok=True)


def normalize_regions(frame):
    frame = frame.copy()
    names = frame["시도명"].astype("string").str.strip()
    frame["시도명"] = names.map(REGION_MAP)
    assert frame["시도명"].notna().all(), f"알 수 없는 시도명: {names[frame['시도명'].isna()].tolist()}"
    assert len(frame) == 17 and frame["시도명"].is_unique, "17개 고유 시도가 필요합니다."
    assert set(frame["시도명"]) == REGIONS, "시도 목록 불일치"
    return frame


def validate(frame):
    normalize_regions(frame)
    assert not frame.isna().any().any(), "결측값이 있습니다. 보간하지 않습니다."
    numeric = frame.drop(columns="시도명").apply(pd.to_numeric, errors="raise")
    assert np.isfinite(numeric.to_numpy()).all(), "무한대 또는 비정상 수치"


def load_dataset(path):
    frame = normalize_regions(pd.read_csv(path, encoding="utf-8-sig"))
    validate(frame)
    return frame


def require_variation(frame):
    assert (frame.nunique() > 1).all(), "상수 변수는 분석할 수 없습니다."


def save_csv(frame, path, index=False):
    frame.to_csv(path, index=index, encoding="utf-8-sig")
