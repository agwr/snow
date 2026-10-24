from typing import Any

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.colors import CenteredNorm

from snow._theme import Alpha, Color, Colormap, Font, LineWidth
from snow.chart import POINT_LABEL, ChartLayer, ColorBy


def scatter(
    data: pl.DataFrame,
    *,
    x: str,
    y: str,
    point_labels: str | None = None,
    y_low: str | None = None,
    y_high: str | None = None,
    color: Color | ColorBy = Color.BLUE,
    **matplotlib_kwargs: Any,
) -> ChartLayer:
    """
    One point per row, with optional text labels and error bars.
    """

    def draw_on(ax: Axes) -> None:
        for group, style in color.groups(data):
            ax.scatter(group[x], group[y], alpha=Alpha.POINT.value, **style, **matplotlib_kwargs)

            if y_low is not None and y_high is not None:
                ax.errorbar(
                    group[x],
                    group[y],
                    yerr=(group[y] - group[y_low], group[y_high] - group[y]),
                    fmt="none",
                    ecolor=style["color"],
                    alpha=Alpha.POINT.value,
                    elinewidth=LineWidth.ERROR_BAR.value,
                    capsize=3,
                    capthick=LineWidth.ERROR_BAR.value,
                    zorder=0.9,
                )

        if point_labels is not None:
            for label, point_x, point_y in zip(data[point_labels], ax.convert_xunits(data[x]), data[y]):
                if label is not None:
                    ax.text(point_x, point_y, str(label), fontsize=Font.POINT_LABEL_SIZE.value, gid=POINT_LABEL)

    return ChartLayer(draw_on=draw_on)


def line(
    data: pl.DataFrame,
    *,
    x: str,
    y: str,
    y_low: str | None = None,
    y_high: str | None = None,
    color: Color | ColorBy = Color.BLUE,
    **matplotlib_kwargs: Any,
) -> ChartLayer:
    """
    One line, or one per category when `color` is a `ColorBy`, with an optional shaded band from `y_low` to `y_high`.
    """

    def draw_on(ax: Axes) -> None:
        for group, style in color.groups(data):
            ax.plot(group[x], group[y], linewidth=LineWidth.LINE.value, **style, **matplotlib_kwargs)

            if y_low is not None and y_high is not None:
                ax.fill_between(
                    group[x], group[y_low], group[y_high], color=style["color"], alpha=Alpha.BAND.value, linewidth=0
                )

    return ChartLayer(draw_on=draw_on)


def histogram(
    data: pl.DataFrame,
    *,
    x: str,
    bins: int = 50,
    density: bool = False,
    color: Color | ColorBy = Color.BLUE,
    **matplotlib_kwargs: Any,
) -> ChartLayer:
    """
    The distribution of `x`, or one per category when `color` is a `ColorBy`.
    """

    def draw_on(ax: Axes) -> None:
        edges = np.histogram_bin_edges(data[x], bins=bins)

        for group, style in color.groups(data):
            ax.hist(group[x], bins=edges, density=density, alpha=Alpha.HISTOGRAM.value, **style, **matplotlib_kwargs)

    return ChartLayer(draw_on=draw_on)


def boxplot(data: pl.DataFrame, *, x: str, y: str, color: Color = Color.BLUE, **matplotlib_kwargs: Any) -> ChartLayer:
    """
    One box per category of `x`, summarizing the distribution of `y`.
    """

    def draw_on(ax: Axes) -> None:
        groups = data.partition_by(x, maintain_order=True, as_dict=True)

        ax.boxplot(
            [group[y] for group in groups.values()],
            tick_labels=[str(category) for (category) in groups],
            patch_artist=True,
            boxprops={"facecolor": color.value},
            medianprops={"color": Font.COLOR.value},
            **matplotlib_kwargs,
        )

    return ChartLayer(draw_on=draw_on)


def heatmap(
    values: np.ndarray,
    *,
    row_tick_labels: list[str] | None = None,
    column_tick_labels: list[str] | None = None,
    midpoint: float | None = None,
    hidden_cells: np.ndarray | None = None,
    colorbar_label: str | None = None,
    **matplotlib_kwargs: Any,
) -> ChartLayer:
    """
    A 2D array as colored cells. `midpoint` selects the diverging colormap.
    """

    def draw_on(ax: Axes) -> None:
        colormap, norm = (
            (Colormap.SEQUENTIAL, None) if midpoint is None else (Colormap.DIVERGING, CenteredNorm(vcenter=midpoint))
        )

        image = ax.imshow(
            np.ma.masked_array(values, mask=hidden_cells),
            cmap=colormap.value,
            norm=norm,
            aspect="auto",
            **matplotlib_kwargs,
        )
        ax.figure.colorbar(image, ax=ax, label=colorbar_label)

        if row_tick_labels is not None:
            ax.set_yticks(range(len(row_tick_labels)), row_tick_labels)

        if column_tick_labels is not None:
            ax.set_xticks(range(len(column_tick_labels)), column_tick_labels, rotation=90)

    return ChartLayer(draw_on=draw_on)
