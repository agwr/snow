from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from enum import Enum, auto
from itertools import cycle, pairwise
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from adjustText import adjust_text
from matplotlib import font_manager, rc_context
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import STEP_LOOKUP_MAP

from snow._theme import MATPLOTLIB_SETTINGS, Color, Font, LineWidth

POINT_LABEL: Final = "snow.point_label"


class Layout(Enum):
    CARTESIAN = auto()
    CATEGORICAL = auto()
    HEATMAP = auto()


class LegendCorner(Enum):
    UPPER_LEFT = "upper left"
    UPPER_RIGHT = "upper right"
    LOWER_LEFT = "lower left"
    LOWER_RIGHT = "lower right"


class Objective(Enum):
    MINIMIZE = auto()
    MAXIMIZE = auto()


class Scale(Enum):
    LINEAR = "linear"
    LOG = "log"
    SYMLOG = "symlog"


@dataclass(frozen=True)
class ChartLayer:
    """
    A description of one thing to draw. Nothing is drawn until `render` calls `draw_on` with a plot area.
    """

    draw_on: Callable[[Axes], None]


@dataclass(frozen=True)
class ColorBy:
    """
    A different color for each category of `column`. `colors` fixes the color of the listed categories;
    the rest follow palette order.
    """

    column: str
    colors: Mapping[Any, Color] | None = None

    def groups(self, data: pl.DataFrame) -> Iterator[tuple[pl.DataFrame, dict[str, Any]]]:
        """
        One group of rows per category of `column`, in order of first appearance, with its color and label.
        """

        palette = cycle(Color)
        for (category), group in data.group_by(self.column, maintain_order=True):
            assigned = (self.colors or {}).get(category) or next(palette)
            yield group, {"color": assigned.value, "label": str(category)}


@dataclass(frozen=True, kw_only=True)
class Axis:
    """
    The label, scale, and range of one axis. `limits` is in data units: numbers, or dates on a date axis.
    """

    label: str | None = None
    scale: Scale = Scale.LINEAR
    limits: tuple[float, float] | tuple[date, date] | None = None


@dataclass(frozen=True, kw_only=True)
class Panel:
    """
    A description of one plot area: its layers and its y-axis.
    Layers are drawn in order, so later layers sit on top.
    """

    layers: tuple[ChartLayer, ...]
    y_axis: Axis = Axis()


@dataclass(frozen=True, kw_only=True)
class Chart:
    """
    A description of a whole chart: panels stacked top to bottom on one shared x-axis, and the text
    around them. The legend is drawn on the top panel only. Nothing is drawn until `render`.
    """

    panels: tuple[Panel, ...]
    layout: Layout = Layout.CARTESIAN
    title: str | None = None
    x_axis: Axis = Axis()
    caption: str | None = None
    legend_corner: LegendCorner = LegendCorner.UPPER_LEFT


def _place_point_labels(ax: Axes) -> None:
    labels = [text for text in ax.texts if text.get_gid() == POINT_LABEL]
    if not labels:
        return

    points = [
        collection.get_offset_transform().transform(collection.get_offsets())
        for collection in ax.collections
        if isinstance(collection, PathCollection)
    ]
    lines = [
        line.get_transform().transform(np.column_stack(STEP_LOOKUP_MAP[line.get_drawstyle()](*line.get_data(False))))
        for line in ax.lines
        if line.get_linestyle() != "None"
    ]
    traced_lines = [
        np.linspace(start, end, max(1, int(np.hypot(*(end - start)) / 8)), endpoint=False)
        for line in lines
        for start, end in pairwise(line[np.isfinite(line).all(axis=1)])
    ]
    obstacles = np.vstack([np.empty((0, 2)), *points, *traced_lines])
    obstacles = ax.transData.inverted().transform(obstacles[np.isfinite(obstacles).all(axis=1)])

    adjust_text(
        labels,
        x=obstacles[:, 0],
        y=obstacles[:, 1],
        ax=ax,
        expand=(1.2, 1.5),
        force_text=(0.5, 0.8),
        force_pull=(0.005, 0.005),
        iter_lim=300,
        min_arrow_len=15,
        arrowprops={"arrowstyle": "-", "color": Font.COLOR.value, "linewidth": LineWidth.LEADER_LINE.value},
    )


def render(chart: Chart) -> Figure:
    """
    Create a new matplotlib figure, call each layer's `draw_on` in order, and place all text by the layout.
    """

    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    if Font.FAMILY.value not in installed_fonts:
        font_directories = [*font_manager.X11FontDirectories, *font_manager.OSXFontDirectories]
        for font_path in font_manager.findSystemFonts(font_directories):
            if Path(font_path).name.startswith(f"{Font.FAMILY.value}-"):
                font_manager.fontManager.addfont(font_path)

    with rc_context(MATPLOTLIB_SETTINGS):
        width = 10 if chart.layout is Layout.HEATMAP else 13
        figure = Figure(figsize=(width, 4 + 4 * len(chart.panels)), layout="constrained")
        axes = figure.subplots(len(chart.panels), sharex=True, squeeze=False)[:, 0]
        top, bottom = axes[0], axes[-1]

        bottom.set_xscale(chart.x_axis.scale.value)
        if chart.x_axis.limits is not None:
            bottom.set_xlim(chart.x_axis.limits)
        legend_entries: dict[str, Artist] = {}

        for ax, panel in zip(axes, chart.panels):
            match chart.layout:
                case Layout.CATEGORICAL:
                    ax.grid(False, axis="x")
                case Layout.HEATMAP:
                    ax.grid(False)

            ax.set_yscale(panel.y_axis.scale.value)
            if panel.y_axis.limits is not None:
                ax.set_ylim(panel.y_axis.limits)
            for layer in panel.layers:
                layer.draw_on(ax)
            ax.set_ylabel(panel.y_axis.label)

            for handle, label in zip(*ax.get_legend_handles_labels()):
                legend_entries.setdefault(label, handle)

        top.set_title(chart.title)
        bottom.set_xlabel(chart.x_axis.label)

        if legend_entries:
            top.legend(legend_entries.values(), legend_entries.keys(), loc=chart.legend_corner.value)

        if chart.caption is not None:
            bottom.annotate(
                chart.caption,
                xy=(0, 0),
                xycoords=("axes fraction", bottom.xaxis.label),
                xytext=(0, -8),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=Font.CAPTION_SIZE.value,
            )

        figure.draw_without_rendering()
        for ax in axes:
            _place_point_labels(ax)
        figure.draw_without_rendering()

    return figure


def save(chart: Chart, path: str | Path, **matplotlib_kwargs: Any) -> Path:
    """
    Render the chart, write it to `path`, and return the resolved path.
    """

    path = Path(path).resolve()

    with rc_context(MATPLOTLIB_SETTINGS):
        render(chart).savefig(path, **matplotlib_kwargs)

    return path
