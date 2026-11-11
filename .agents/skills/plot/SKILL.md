---
name: plot
description: API reference and tips for snow, a plotting library for machine learning research results.
---

# Snow

Snow is a plotting library for machine learning research results. It uses matplotlib. The style and the layout are fixed. Plots take Polars DataFrames or NumPy arrays. Each signature shows the type.

A chart has three parts:

1. A layer draws one thing, for example points or a line.
2. A panel holds layers and a y-axis.
3. A chart holds panels, one shared x-axis, and the text around them.

The functions `render` and `save` draw the chart. The other functions only describe it.

```python
import polars as pl
import snow

runs = pl.DataFrame({"step": [0, 100, 200, 300], "loss": [3.1, 2.4, 2.0, 1.8]})

curve = snow.line(runs, x="step", y="loss")
panel = snow.Panel(layers=(curve,), y_axis=snow.Axis(label="Loss"))
chart = snow.Chart(panels=(panel,), title="Training Loss", x_axis=snow.Axis(label="Step"))

snow.save(chart, "loss.png")
```

## API reference

All names come from `import snow`. Read the signatures and docstrings in the `snow` package for the fields and the defaults. The examples use this data:

```python
from datetime import date

import numpy as np
import polars as pl
import snow

scores = pl.DataFrame(
    {
        "model": ["A-1", "B-1", "A-2", "C-1", "B-2"],
        "lab": ["Lab A", "Lab B", "Lab A", "Lab C", "Lab B"],
        "release_date": [date(2025, 1, 10), date(2025, 3, 2), date(2025, 5, 20), date(2025, 7, 1), date(2025, 9, 15)],
        "score": [41.0, 47.5, 52.0, 49.0, 58.5],
        "score_low": [38.0, 44.0, 49.5, 45.0, 55.0],
        "score_high": [44.0, 51.0, 54.5, 53.0, 62.0],
        "cost": [1.2, 3.5, 2.0, 0.8, 6.0],
    }
)
runs = pl.DataFrame(
    {
        "step": [0, 100, 200, 300] * 2,
        "loss": [3.1, 2.4, 2.0, 1.8, 3.0, 2.2, 1.9, 1.6],
        "loss_low": [3.0, 2.3, 1.9, 1.7, 2.9, 2.1, 1.8, 1.5],
        "loss_high": [3.2, 2.5, 2.1, 1.9, 3.1, 2.3, 2.0, 1.7],
        "run": ["baseline"] * 4 + ["treatment"] * 4,
    }
)
```

### Descriptions

`Chart` is the whole chart. Panels stack from top to bottom and share the x-axis. The legend shows once and collects the entries of all panels.

`Panel` is one plot area. The layers draw in order, so a later layer shows on top of an earlier layer.

`Axis` holds the label, the scale, and the range of one axis.

`ChartLayer` is one thing to draw. It wraps a function that takes a matplotlib `Axes`. All plot functions and overlay functions return one.

```python
curves = snow.line(runs, x="step", y="loss", color=snow.ColorBy("run"))
panel = snow.Panel(layers=(curves,), y_axis=snow.Axis(label="Loss", scale=snow.Scale.LOG))
chart = snow.Chart(panels=(panel,), title="Training Loss", x_axis=snow.Axis(label="Step"))
target = snow.ChartLayer(draw_on=lambda ax: ax.axhspan(1.5, 2.0, color=snow.Color.GREEN.value, alpha=0.1))
```

### Options

`Layout` sets the figure type of a chart. Use the heatmap layout with `heatmap` and the categorical layout with `boxplot`.

`Scale` sets the scale of an axis.

`LegendCorner` sets the corner that holds the legend.

`Objective` sets the better direction on an axis for `frontier`.

```python
x_axis = snow.Axis(label="Cost ($)", scale=snow.Scale.LOG)
chart = snow.Chart(panels=(panel,), legend_corner=snow.LegendCorner.UPPER_RIGHT)
```

### Colors

`Color` is the fixed palette. A plot with one `Color` draws all rows in that color.

`ColorBy` gives one color to each category of a column and one legend entry to each category. It can fix the color of some categories.

```python
points = snow.scatter(scores, x="release_date", y="score", color=snow.Color.RED)
lab_colors = snow.ColorBy("lab", colors={"Lab A": snow.Color.BLUE, "Lab B": snow.Color.ORANGE})
points = snow.scatter(scores, x="release_date", y="score", color=lab_colors)
```

### Plots

Each plot returns a layer. Extra keyword arguments go to the matplotlib call under the plot.

`scatter` draws one point for each row. It can add error bars and text labels.

```python
points = snow.scatter(
    scores,
    x="release_date",
    y="score",
    point_labels="model",
    y_low="score_low",
    y_high="score_high",
    color=snow.ColorBy("lab"),
)
```

`line` draws one line, or one line for each category. It can add a shaded band.

```python
curves = snow.line(runs, x="step", y="loss", y_low="loss_low", y_high="loss_high", color=snow.ColorBy("run"))
```

`histogram` draws a distribution, or one distribution for each category, on shared bins.

```python
counts = snow.histogram(runs, x="loss", bins=10, color=snow.ColorBy("run"))
```

`boxplot` draws one box for each category.

```python
boxes = snow.boxplot(runs, x="run", y="loss")
chart = snow.Chart(panels=(snow.Panel(layers=(boxes,)),), layout=snow.Layout.CATEGORICAL)
```

`heatmap` draws a 2D array as colored cells with a colorbar. A midpoint gives a diverging colormap. A boolean mask hides cells.

```python
tokens = ["<bos>", "the", "cat", "sat"]
logits = np.random.default_rng(0).normal(size=(4, 4))
causal_mask = np.triu(np.ones((4, 4), dtype=bool), k=1)

cells = snow.heatmap(logits, row_tick_labels=tokens, column_tick_labels=tokens, midpoint=0, hidden_cells=causal_mask)
chart = snow.Chart(panels=(snow.Panel(layers=(cells,)),), layout=snow.Layout.HEATMAP)
```

### Overlays

Each overlay returns a layer that draws over the data.

`trendline` draws the least-squares line with a confidence band. It can add a prediction band.

`frontier` draws a dashed staircase through the Pareto frontier.

`horizontal_line` and `vertical_line` draw a thin guide line across the panel.

```python
fit = snow.trendline(scores, x="release_date", y="score")
record = snow.frontier(scores, x="release_date", y="score")
target = snow.horizontal_line(50)
launch = snow.vertical_line(date(2025, 6, 1))
```

### Output

`render` draws the chart into a new matplotlib figure and returns it.

`save` draws the chart and writes it to a file. The file extension sets the format.

```python
figure = snow.render(chart)
path = snow.save(chart, "chart.pdf")
```

## Tips

- `trendline` and `frontier` fade only the scatter points in earlier layers. Put them after the data layers.
- Each `ColorBy` assigns palette colors independently. Use the same `colors` mapping in each `ColorBy` that must match.
- Categories and legend entries follow the order of first appearance in the data. Sort the data to set the order.
- Legend entries come only from `ColorBy`.
- Label a subset of points with a label column that holds null for the other rows. Many labels make placement slow. This expression labels each model that sets a new record on its release date:
  ```python
  scores = scores.sort("release_date").with_columns(
      record=pl.when(pl.col("score") > pl.col("score").cum_max().shift(1, fill_value=float("-inf"))).then("model")
  )
  points = snow.scatter(scores, x="release_date", y="score", point_labels="record")
  ```
- The default objectives of `frontier` trace the running best score on a date axis.
- Set the layout for each heatmap and each boxplot. The default layout silently draws grid lines through cells and categories.
