"""07~10 그림 공통 설정: 한글 글꼴, 색상, 시도 약칭."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from utils import REGION_MAP

# 02·06과 같은 우선순위로 설치된 한글 글꼴을 찾는다.
KOREAN_FONTS = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]

# 보고서 그림 색상
BLUE = "#2A7AB8"          # 그림 1 평균 이하, 그림 2·3 보유 수준·막대
BLUE_LIGHT = "#8DBAD8"
ORANGE = "#E07B22"        # 그림 1 평균 초과
ORANGE_LIGHT = "#EDBC8A"
DECOMP_COLORS = {"보유": "#2F7ED8", "주행": "#F26B3A", "강도": "#1DB386"}
GRID = "#E3E3E3"
AXIS = "#555555"

# 서울특별시 -> 서울 (utils.REGION_MAP의 두 글자 약칭을 역으로 사용)
SHORT_NAME = {full: short for short, full in REGION_MAP.items() if len(short) == 2}


def use_korean_font():
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font in KOREAN_FONTS:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    else:
        raise RuntimeError("한글 글꼴을 설치하세요: Malgun Gothic 또는 NanumGothic 등")
    plt.rcParams["axes.unicode_minus"] = False


def clean_axes(ax, keep_bottom=True):
    """위·오른쪽·왼쪽 테두리를 지우고 세로 격자만 옅게 남긴다."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(keep_bottom)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
