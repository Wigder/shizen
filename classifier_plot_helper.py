import pickle
from os.path import basename

import plotly.graph_objs as go
import plotly.offline as py

smoothing = 1.3


def plot_history(title, history_path, y_range=None):
    name = basename(history_path).split(".pickle")[0]
    with open(history_path, "rb") as f:
        history = pickle.load(f)

    train_acc = history.history["acc"]
    train_loss = history.history["loss"]
    val_acc = history.history["val_acc"]
    val_loss = history.history["val_loss"]
    epochs = [e + 1 for e in history.epoch]

    train_acc_trace = go.Scatter(
        x=epochs,
        y=train_acc,
        mode="lines",
        name="Training",
        line=dict(
            shape="spline",
            smoothing=smoothing,
            color="steelblue"
        )
    )
    val_acc_trace = go.Scatter(
        x=epochs,
        y=val_acc,
        mode="lines",
        name="Validation",
        line=dict(
            shape="spline",
            smoothing=smoothing,
            color="firebrick"
        )
    )
    layout_acc = go.Layout(
        title="Accuracy Throughout Training ({})".format(title),
        xaxis=dict(
            range=[epochs[0], epochs[-1]],
            title="Epoch",
            showline=True
        ),
        yaxis=dict(
            title="Accuracy",
            showline=True
        ),
        font=dict(family="Calibri"),
        showlegend=True
    )
    fig_acc = go.Figure(data=[train_acc_trace, val_acc_trace], layout=layout_acc)
    py.plot(fig_acc, filename="out/{}_acc.html".format(name))

    train_loss_trace = go.Scatter(
        x=epochs,
        y=train_loss,
        mode="lines",
        name="Training",
        line=dict(
            shape="spline",
            smoothing=smoothing,
            color="steelblue"
        )
    )
    val_loss_trace = go.Scatter(
        x=epochs,
        y=val_loss,
        mode="lines",
        name="Validation",
        line=dict(
            shape="spline",
            smoothing=smoothing,
            color="firebrick"
        )
    )
    layout_loss = go.Layout(
        title="Loss Throughout Training ({})".format(title),
        xaxis=dict(
            range=[epochs[0], epochs[-1]],
            title="Epoch",
            showline=True
        ),
        yaxis=dict(
            range=y_range,
            title="Loss",
            showline=True
        ),
        font=dict(family="Calibri"),
        showlegend=True
    )
    fig_loss = go.Figure(data=[train_loss_trace, val_loss_trace], layout=layout_loss)
    py.plot(fig_loss, filename="out/{}_loss.html".format(name))
