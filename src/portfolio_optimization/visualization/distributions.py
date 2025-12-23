import numpy as np
import pandas as pd
from typing import Tuple
from scipy.stats import gaussian_kde, multivariate_normal
import plotly.graph_objects as go

def plot_3d_kde_surface(
    returns_df: pd.DataFrame,
    assets: Tuple[str, str],
    *,
    bw_method: str | float = "scott",
    grid_n: int = 120,
    pad: float = 0.05,
) -> None:
    """
    Plot a true 3D bell surface (x, y, density) using a Gaussian KDE
    for exactly two assets.
    """
    if len(assets) != 2:
        raise ValueError("Must provide exactly 2 assets to plot a 3D KDE surface.")

    Y = returns_df[list(assets)].dropna().to_numpy()   # (n, 2)
    if Y.shape[0] < 5:
        raise ValueError("Not enough observations to fit KDE.")

    # KDE expects (d, n)
    X = Y.T
    kde = gaussian_kde(X, bw_method=bw_method)

    # grid using full data range (no quantiles)
    x_min, y_min = Y.min(axis=0)
    x_max, y_max = Y.max(axis=0)

    # small padding so the surface doesn't clip
    dx = pad * (x_max - x_min)
    dy = pad * (y_max - y_min)

    x = np.linspace(x_min - dx, x_max + dx, grid_n)
    y = np.linspace(y_min - dy, y_max + dy, grid_n)
    Xg, Yg = np.meshgrid(x, y, indexing="xy")

    grid_points = np.vstack([Xg.ravel(), Yg.ravel()])
    Z = kde(grid_points).reshape(grid_n, grid_n)

    fig = go.Figure(data=go.Surface(x=Xg, y=Yg, z=Z))
    fig.update_layout(
        title="3D Bell Curve (2-Asset Gaussian KDE)",
        scene=dict(
            xaxis_title=assets[0],
            yaxis_title=assets[1],
            zaxis_title="Density",
        ),
    )
    fig.show()


def plot_3d_gaussian_surface(
    returns_df: pd.DataFrame,
    assets: Tuple[str, str],
    *,
    grid_n: int = 120,
    pad: float = 0.05,
) -> None:
    """
    Plot a true 3D bell surface (x, y, density) for a fitted
    2D multivariate normal.
    """
    if len(assets) != 2:
        raise ValueError("Must provide exactly 2 assets to plot a 3D Gaussian surface.")

    Y = returns_df[list(assets)].dropna().to_numpy()   # (n, 2)
    if Y.shape[0] < 3:
        raise ValueError("Not enough observations to fit Gaussian.")

    mu = Y.mean(axis=0)
    Sigma = np.cov(Y, rowvar=False)
    rv = multivariate_normal(mean=mu, cov=Sigma)

    # grid using full data range (no quantiles)
    x_min, y_min = Y.min(axis=0)
    x_max, y_max = Y.max(axis=0)

    dx = pad * (x_max - x_min)
    dy = pad * (y_max - y_min)

    x = np.linspace(x_min - dx, x_max + dx, grid_n)
    y = np.linspace(y_min - dy, y_max + dy, grid_n)
    Xg, Yg = np.meshgrid(x, y, indexing="xy")

    pos = np.dstack((Xg, Yg))
    Z = rv.pdf(pos)

    fig = go.Figure(data=go.Surface(x=Xg, y=Yg, z=Z))
    fig.update_layout(
        title="3D Bell Curve (2-Asset Multivariate Gaussian)",
        scene=dict(
            xaxis_title=assets[0],
            yaxis_title=assets[1],
            zaxis_title="Density",
        ),
    )
    fig.show()


def plot_multivariate_returns(
    returns_df: pd.DataFrame,
    assets: Tuple[str, str],
    type: str
) -> None:
    if len(assets) != 2:
        raise ValueError("Must provide exactly 2 assets to plot 3D KDE.")
    
    if type == "KDE":
        print("Plotting 3D Guassian KDE of multivariate returns...")
        plot_3d_kde_surface(returns_df, assets)
    
    elif type == "Gaussian":
        print("Plotting Multivariate Normal (Gaussian) of multivariate returns...")
        plot_3d_gaussian_surface(returns_df, assets)
        