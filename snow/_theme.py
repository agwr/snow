from collections.abc import Iterator, Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

import polars as pl
from cycler import cycler


class Color(Enum):
    BLUE = "#0173B2"
    ORANGE = "#DE8F05"
    GREEN = "#029E73"
    RED = "#D55E00"
    PURPLE = "#CC78BC"
    BROWN = "#CA9161"
    PINK = "#FBAFE4"
    GRAY = "#949494"
    YELLOW = "#ECE133"
    CYAN = "#56B4E9"

    def groups(self, data: pl.DataFrame) -> Iterator[tuple[pl.DataFrame, dict[str, Any]]]:
        yield data, {"color": self.value}


class Font(Enum):
    FAMILY = "Ubuntu"
    COLOR = "#000000"
    TICK_SIZE = 11
    CAPTION_SIZE = 10.5
    POINT_LABEL_SIZE = 9


class LineWidth(Enum):
    LINE = 2.0
    TRENDLINE = 2.2
    ERROR_BAR = 1.0
    GUIDE_LINE = 1.0
    LEADER_LINE = 0.6


class Alpha(Enum):
    POINT = 0.45
    POINT_UNDER_OVERLAY = 0.3
    BAND = 0.16
    HISTOGRAM = 0.5
    GUIDE_LINE = 0.6


class Colormap(Enum):
    SEQUENTIAL = "viridis"
    DIVERGING = "RdBu_r"


MATPLOTLIB_SETTINGS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "axes.facecolor": "#EAEAF2",
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.left": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.spines.bottom": False,
        "axes.prop_cycle": cycler(color=[color.value for color in Color]),
        "lines.markersize": 8,
        "grid.color": "white",
        "text.color": Font.COLOR.value,
        "axes.labelcolor": Font.COLOR.value,
        "xtick.color": Font.COLOR.value,
        "ytick.color": Font.COLOR.value,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "font.family": Font.FAMILY.value,
        "axes.titlesize": 18,
        "axes.labelsize": 12,
        "xtick.labelsize": Font.TICK_SIZE.value,
        "ytick.labelsize": Font.TICK_SIZE.value,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "savefig.dpi": 300,
    }
)
