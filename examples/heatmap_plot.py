from pathlib import Path

import numpy as np

import snow

# Tokenize a sequence with repeated phrases, so the head can copy from earlier occurrences
tokens = (
    "<bos> the cat sat on the mat and the dog sat on the rug . "
    "then the cat saw the dog , and the dog saw the cat . "
    "the cat sat on the rug and the dog sat on the mat ."
).split()
words = np.array(tokens)
length = len(tokens)
queries, keys = np.indices((length, length))

# Synthesize logits as a weighted sum of common attention patterns
sink = 2.5 * (keys == 0)
previous_token = 3.0 * (keys == queries - 1)
recency = -0.08 * (queries - keys)
induction = 4.0 * np.pad(words[:, None] == words[None, :-1], ((0, 0), (1, 0)))
noise = np.random.default_rng(0).normal(scale=0.6, size=(length, length))
logits = sink + previous_token + recency + induction + noise
causal_mask = keys > queries

# Plot the logits
cells = snow.heatmap(
    logits,
    row_tick_labels=tokens,
    column_tick_labels=tokens,
    midpoint=0,
    hidden_cells=causal_mask,
    colorbar_label="Logit",
)

# Build the chart with a single panel
panel = snow.Panel(layers=(cells,), y_axis=snow.Axis(label="Query token"))
chart = snow.Chart(
    panels=(panel,),
    layout=snow.Layout.HEATMAP,
    title="Attention Logits, Layer 12 Head 7",
    x_axis=snow.Axis(label="Key token"),
)

print(snow.save(chart, Path(__file__).with_suffix(".png")))
