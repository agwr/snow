from datetime import date
from pathlib import Path
from urllib.request import urlopen

import polars as pl

import snow

# Color the major labs, and group the rest as "Other"
lab_colors = {
    "Anthropic": snow.Color.BLUE,
    "Google DeepMind": snow.Color.ORANGE,
    "OpenAI": snow.Color.GREEN,
    "xAI": snow.Color.RED,
    "Meta AI": snow.Color.PURPLE,
    "Alibaba": snow.Color.BROWN,
    "DeepSeek": snow.Color.CYAN,
    "Other": snow.Color.GRAY,
}

# Fetch ECI scores and their 90% confidence intervals from Epoch AI, for models released since 2025
scores = (
    pl.read_csv(urlopen("https://epoch.ai/data/eci_scores.csv").read(), try_parse_dates=True)
    .filter(pl.col("date") >= date(2025, 1, 1))
    .select(
        model_name="Model",
        release_date="date",
        eci="eci",
        eci_low="eci_ci_low",
        eci_high="eci_ci_high",
        lab=pl.when(pl.col("Organization").is_in(list(lab_colors))).then("Organization").otherwise(pl.lit("Other")),
    )
)

# Plot the scores with their confidence intervals, and the record frontier on top
points = snow.scatter(
    scores, x="release_date", y="eci", y_low="eci_low", y_high="eci_high", color=snow.ColorBy("lab", colors=lab_colors)
)
frontier = snow.frontier(scores, x="release_date", y="eci")

# Build the chart with a single panel
panel = snow.Panel(layers=(points, frontier), y_axis=snow.Axis(label="ECI"))
chart = snow.Chart(panels=(panel,), title="Epoch Capabilities Index Over Time", x_axis=snow.Axis(label="Release Date"))

print(snow.save(chart, Path(__file__).with_suffix(".png")))
