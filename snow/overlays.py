from typing import Any

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from scipy.stats import t

from snow._theme import Alpha, Color, LineWidth
from snow.chart import ChartLayer, Objective


def _convert_x_to_floats(ax: Axes, data: pl.DataFrame, x: str) -> np.ndarray:
    ax.xaxis.update_units(data[x])
    return np.asarray(ax.convert_xunits(data[x]), dtype=float)


def trendline(
    data: pl.DataFrame,
    *,
    x: str,
    y: str,
    ci: float | None = 0.95,
    pi: float | None = None,
    color: Color = Color.BLUE,
    **matplotlib_kwargs: Any,
) -> ChartLayer:
    """
    The least-squares line through the points, with a CI band and an optional prediction interval.
    Fades the scatter points drawn before it.
    """

    def draw_on(ax: Axes) -> None:
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                collection.set_alpha(Alpha.POINT_UNDER_OVERLAY.value)

        xs, ys = _convert_x_to_floats(ax, data, x), data[y].to_numpy()
        slope, intercept = np.polyfit(xs, ys, 1)
        grid = np.linspace(xs.min(), xs.max(), 200)
        fitted = intercept + slope * grid

        ax.plot(grid, fitted, color=color.value, linewidth=LineWidth.TRENDLINE.value, **matplotlib_kwargs)

        n = len(xs)
        residual_std = np.sqrt(np.sum((ys - intercept - slope * xs) ** 2) / (n - 2))
        leverage = 1 / n + (grid - xs.mean()) ** 2 / np.sum((xs - xs.mean()) ** 2)

        for level, noise in ((ci, 0), (pi, 1)):
            if level is not None:
                half_width = t.ppf((1 + level) / 2, n - 2) * residual_std * np.sqrt(noise + leverage)

                ax.fill_between(
                    grid,
                    fitted - half_width,
                    fitted + half_width,
                    color=color.value,
                    alpha=Alpha.BAND.value,
                    linewidth=0,
                    zorder=0.8,
                )

    return ChartLayer(draw_on=draw_on)


def frontier(
    data: pl.DataFrame,
    *,
    x: str,
    y: str,
    x_objective: Objective = Objective.MINIMIZE,
    y_objective: Objective = Objective.MAXIMIZE,
    color: Color = Color.GRAY,
    **matplotlib_kwargs: Any,
) -> ChartLayer:
    """
    The dashed staircase through the points that no other point beats on both x and y.
    Fades the scatter points drawn before it.
    """

    def draw_on(ax: Axes) -> None:
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                collection.set_alpha(Alpha.POINT_UNDER_OVERLAY.value)

        xs, ys = _convert_x_to_floats(ax, data, x), data[y].to_numpy()

        x_cost = xs if x_objective is Objective.MINIMIZE else -xs
        y_score = ys if y_objective is Objective.MAXIMIZE else -ys

        order = np.lexsort((-y_score, x_cost))
        best_before = np.concatenate(([-np.inf], np.maximum.accumulate(y_score[order])[:-1]))
        on_frontier = order[y_score[order] > best_before]

        ax.step(
            xs[on_frontier],
            ys[on_frontier],
            where="post",
            color=color.value,
            linewidth=LineWidth.LINE.value,
            linestyle="--",
            **matplotlib_kwargs,
        )

    return ChartLayer(draw_on=draw_on)


def horizontal_line(y: float, *, color: Color = Color.GRAY, **matplotlib_kwargs: Any) -> ChartLayer:
    """
    A dotted line across the plot area at `y`.
    """

    def draw_on(ax: Axes) -> None:
        ax.axhline(
            y,
            color=color.value,
            linestyle=":",
            linewidth=LineWidth.GUIDE_LINE.value,
            alpha=Alpha.GUIDE_LINE.value,
            zorder=0.9,
            **matplotlib_kwargs,
        )

    return ChartLayer(draw_on=draw_on)


def vertical_line(x: float, *, color: Color = Color.GRAY, **matplotlib_kwargs: Any) -> ChartLayer:
    """
    A dotted line across the plot area at `x`.
    """

    def draw_on(ax: Axes) -> None:
        ax.axvline(
            x,
            color=color.value,
            linestyle=":",
            linewidth=LineWidth.GUIDE_LINE.value,
            alpha=Alpha.GUIDE_LINE.value,
            zorder=0.9,
            **matplotlib_kwargs,
        )

    return ChartLayer(draw_on=draw_on)
