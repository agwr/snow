import json
from pathlib import Path
from urllib.request import urlopen

import polars as pl

import snow

# Fetch release dates from Epoch AI
releases = pl.read_csv(urlopen("https://epoch.ai/data/eci_scores.csv").read(), try_parse_dates=True).select(
    model_name="Model",
    release_date="date",
    lab="Organization",
    slug=pl.col("Model").str.to_lowercase().str.strip_prefix("claude ").str.replace_all(" ", "-"),
)

# Fetch PostTrainBench scores
text = urlopen("https://posttrainbench.com/scores-v1.2.js").read().decode()
aggregated = json.loads(text[text.index("{") : text.rindex("}") + 1])["aggregatedScores"]
scores = (
    pl.DataFrame({"slug": list(aggregated), "score": [entry["avg"] for entry in aggregated.values()]})
    .with_columns(pl.col("slug").str.replace(r"-(x?high|max)$", ""))
    .join(releases, on="slug")
    .group_by("model_name", "release_date", "lab", maintain_order=True)
    .agg(pl.col("score").max())
)

# Color each model by its lab
lab_colors = snow.ColorBy(
    "lab",
    colors={
        "Anthropic": snow.Color.BLUE,
        "Google DeepMind": snow.Color.ORANGE,
        "OpenAI": snow.Color.GREEN,
        "xAI": snow.Color.RED,
        "Z.ai (Zhipu AI)": snow.Color.BROWN,
        "Moonshot": snow.Color.PINK,
    },
)

# Plot the scores with a linear fit on top
points = snow.scatter(scores, x="release_date", y="score", point_labels="model_name", color=lab_colors)
fit = snow.trendline(scores, x="release_date", y="score")

# Build the chart with a single panel
panel = snow.Panel(layers=(points, fit), y_axis=snow.Axis(label="PostTrainBench Score (%)"))
chart = snow.Chart(panels=(panel,), title="PostTrainBench Scores Over Time", x_axis=snow.Axis(label="Release Date"))

print(snow.save(chart, Path(__file__).with_suffix(".png")))
