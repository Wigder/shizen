import pickle
from collections import Counter

import plotly.graph_objs as go
import plotly.offline as py

with open("corpora/political/sanitised/dem_train.txt", encoding="utf-8") as f:
    dem_train = [s.rstrip() for s in f.readlines()]
with open("corpora/political/sanitised/rep_train.txt", encoding="utf-8") as f:
    rep_train = [s.rstrip() for s in f.readlines()]

word_freq = dict(Counter(" ".join(dem_train + rep_train).split()).most_common())

cutoff = 1.96
with open("out/uninformed_z_scores.pickle", "rb") as f:
    uninf = [(w, z) for w, z in pickle.load(f) if abs(z) >= cutoff]

# Full plot
uninf_dem = [(w, z) for w, z in uninf if z > 0]
uninf_rep = [(w, z) for w, z in uninf if z < 0]
dem_trace = go.Scatter(
    x=[word_freq[w] for w, z in uninf_dem],
    y=[z for w, z in uninf_dem],
    name="Democrat",
    mode="markers",
    marker=dict(
        color="steelblue"
    )
)
rep_trace = go.Scatter(
    x=[word_freq[w] for w, z in uninf_rep],
    y=[z for w, z in uninf_rep],
    name="Republican",
    mode="markers",
    marker=dict(
        color="firebrick"
    )
)
layout = go.Layout(
    title="Partisan Words from RtGender Corpus (uninformative Dirichlet prior)",
    xaxis=dict(
        title="Word count",
        type="log",
        showline=True
    ),
    yaxis=dict(
        title="Z-score",
        showline=True
    ),
    font=dict(family="Calibri"),
    showlegend=True
)
fig_sample = go.Figure(data=[dem_trace, rep_trace], layout=layout)
py.plot(fig_sample, filename="out/fightin_words.html")

# Sample plot
uninf_dict = dict(uninf)
"""
Sample words have been extracted randomly and handpicked through trial and error.
"""
uninf_sample_dem = [(w, uninf_dict[w]) for w in
                    ["engel", "popular", "high", "id", "san", "betty", "chellie", "spewing", "make", "electoral"]]
uninf_sample_rep = [(w, uninf_dict[w]) for w in
                    ["grassley", "abide", "pat", "eliminate", "stupid", "christ", "spineless", "mess", "already",
                     "reid"]]

dem_trace_sample = go.Scatter(
    x=[word_freq[w] for w, z in uninf_sample_dem],
    y=[z for w, z in uninf_sample_dem],
    name="Democrat",
    mode="markers+text",
    marker=dict(
        color="steelblue"
    ),
    text=[w for w, z in uninf_sample_dem],
    textposition="middle right",
    textfont=dict(
        size=11
    )
)
rep_trace_sample = go.Scatter(
    x=[word_freq[w] for w, z in uninf_sample_rep],
    y=[z for w, z in uninf_sample_rep],
    name="Republican",
    mode="markers+text",
    marker=dict(
        color="firebrick"
    ),
    text=[w for w, z in uninf_sample_rep],
    textposition="middle right",
    textfont=dict(
        size=11
    )
)
ticks = [-30, -20, -10, -cutoff, 0, cutoff, 10, 20, 30]
layout_sample = go.Layout(
    title="Small Sample of Partisan Words from RtGender Corpus (uninformative Dirichlet prior)",
    xaxis=dict(
        title="Word count",
        type="log",
        showline=True
    ),
    yaxis=dict(
        title="Z-score",
        tickmode="array",
        tickvals=ticks,
        ticktext=[str(i) for i in ticks],
        showline=True
    ),
    shapes=[
        {
            "type": "line",
            "y0": cutoff,
            "y1": cutoff,
            "opacity": 0.5,
            "line": {
                "width": 1,
                "dash": "longdash"
            }
        },
        {
            "type": "line",
            "y0": -cutoff,
            "y1": -cutoff,
            "opacity": 0.5,
            "line": {
                "width": 1,
                "dash": "longdash"
            }
        },
    ],
    showlegend=True,
    font=dict(family="Calibri")
)
fig_sample = go.Figure(data=[dem_trace_sample, rep_trace_sample], layout=layout_sample)
py.plot(fig_sample, filename="out/fightin_words_sample.html")
