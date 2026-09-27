from pathlib import Path

import numpy as np
import pandas as pd

from utils import DATA, TABLES, make_dirs, normalize_regions, save_csv, validate

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main():
    make_dirs()
    # 비용계산근거 시트는 참고자료이므로 읽거나 병합하지 않는다.
    raw = normalize_regions(pd.read_excel(RAW / "시도별_통합데이터_2024.xlsx",
                                          sheet_name="통합데이터", engine="openpyxl"))
    validate(raw)

    def column(prefix, suffix):
        matches = [c for c in raw.columns if c.startswith(prefix) and c.endswith(suffix)]
        assert len(matches) == 1, f"열을 유일하게 식별할 수 없습니다: {prefix}, {suffix}: {matches}"
        return matches[0]

    def values(prefix, suffix):
        return pd.to_numeric(raw[column(prefix, suffix)], errors="raise")

    ages = [column("행정구역_", f"_{age}세" if age < 100 else "_100세 이상")
            for age in range(101)]
    population = raw[ages].apply(pd.to_numeric, errors="raise")
    assert (population >= 0).all().all(), "음수 인구"
    result = raw[["시도명"]].copy()
    result["18세이상인구"] = population.iloc[:, 18:].sum(axis=1, skipna=False)
    result["65세이상인구"] = population.iloc[:, 65:].sum(axis=1, skipna=False)
    result["총인구"] = population.sum(axis=1, skipna=False)
    assert (result[["18세이상인구", "총인구"]] > 0).all().all(), "인구 분모가 0입니다."
    dependent = normalize_regions(pd.read_csv(RAW / "dependent_variable.csv", comment="#",
                                              encoding="utf-8-sig"))
    validate(dependent)
    assert dependent["이용경험률"].between(0, 100).all(), "이용경험률 범위 오류"
    result = result.merge(dependent, on="시도명", validate="one_to_one", how="left")
    result["미경험률"] = 100 - result["이용경험률"]
    days = list("월화수목금토일")
    intervals = raw[[column("배차간격_", f"_{d}_배차간격(분)") for d in days]].apply(pd.to_numeric)
    result["주간평균배차간격"] = intervals.mean(axis=1, skipna=False)
    result["평일평균배차간격"] = intervals.iloc[:, :5].mean(axis=1, skipna=False)
    result["1인당자가용등록대수"] = values("자동차등록대수현황_", "_승용_자가용(대)") / result["총인구"]
    result["1일평균주행거리"] = values("용도별_", "_비사업용_승용차_1일 평균주행거리(㎞/대)")
    result["고령인구비율"] = result["65세이상인구"] / result["18세이상인구"] * 100
    riders = raw[[column("시도별 총 이용인원", f"_{d}_이용인원") for d in days]].apply(pd.to_numeric)
    result["주간평균이용인원"] = riders.mean(axis=1, skipna=False)
    types = {kind: values("유형별 총 이용인원", f"_{kind}_이용인원(명)")
             for kind in ["일반", "어린이", "청소년", "고령자", "기타"]}
    total = pd.DataFrame(types).sum(axis=1, skipna=False)
    assert (total > 0).all(), "월요일 유형별 이용인원 합계가 0입니다."
    # 원자료의 두 집계는 실제로 차이가 있다. 보정 없이 요청한 유형별 합계를 사용한다.
    audit = pd.DataFrame({"시도명": raw["시도명"], "유형별합계": total,
                          "월요일총이용인원": riders.iloc[:, 0], "차이": total - riders.iloc[:, 0]})
    save_csv(audit, TABLES / "monday_total_check.csv")
    if not np.allclose(total, riders.iloc[:, 0], rtol=0, atol=1e-6):
        print("경고: 유형별 합계와 월요일 총이용인원 불일치. 원자료와 요청 계산식을 유지합니다.")
        print(audit.loc[audit["차이"] != 0].to_string(index=False))
    result["월요일고령자비율"] = types["고령자"] / total
    result["월요일일반비율"] = types["일반"] / total
    # 월요일 유형별 비중이 모든 요일에 유사하고 일반+고령자가 성인에 대응한다는
    # 상당히 거친 가정의 근사치다. 실제 성인 고유 이용자 비율로 해석하지 않는다.
    result["성인이용인원추정"] = result["주간평균이용인원"] * (result["월요일일반비율"] + result["월요일고령자비율"])
    result["성인대중교통이용률"] = result["성인이용인원추정"] / result["18세이상인구"]
    result["월평균대중교통비"] = values("시도별_월평균_대중교통비용", "_월평균 비용(만원)")
    validate(result)
    save_csv(result, DATA / "analysis_dataset.csv")
    print(f"분석 데이터 생성: {result.shape}, 결측 0")


if __name__ == "__main__":
    main()
