import numpy as np
import pandas as pd
from typing import Tuple
from scipy.stats import gaussian_kde, multivariate_normal
import plotly.graph_objects as go

def plot_3d_kde(
    returns_df: pd.DataFrame,
    assets: Tuple[str, str, str],
) -> None:
    if len(assets) != 3:
        raise ValueError("Must provide exactly 3 assets to plot 3D KDE.")
    """
    Plot a 3D KDE of returns for exactly three assets.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns (columns = asset names).
    assets : tuple[str, str, str]
        Exactly three asset column names.
    """

    X = returns_df[assets].dropna().to_numpy().T      # shape: (3, n). scipy expects (d, n)

    kde = gaussian_kde(X, bw_method="scott")

    # Build a 3D grid to evaluate density on
    x1 = X[0]; x2 = X[1]; x3 = X[2]
    pad = 0.25  # widen plotting range a bit
    n = 60      # grid resolution (60^3=216k points; increase carefully)

    g1 = np.linspace(x1.min()*(1+pad), x1.max()*(1+pad), n)
    g2 = np.linspace(x2.min()*(1+pad), x2.max()*(1+pad), n)
    g3 = np.linspace(x3.min()*(1+pad), x3.max()*(1+pad), n)

    G1, G2, G3 = np.meshgrid(g1, g2, g3, indexing="ij")
    grid_points = np.vstack([G1.ravel(), G2.ravel(), G3.ravel()])  # (3, n^3)
    D = kde(grid_points).reshape(n, n, n)  # density volume

    # Choose a density threshold to show an isosurface
    # e.g. show top ~5% densest region
    thr = np.quantile(D, 0.95)

    fig = go.Figure(
        data=go.Isosurface(
            x=G1.ravel(),
            y=G2.ravel(),
            z=G3.ravel(),
            value=D.ravel(),
            isomin=thr,
            isomax=D.max(),
            surface_count=2,   # number of nested surfaces
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=assets[0],
            yaxis_title=assets[1],
            zaxis_title=assets[2],
        ),
        title="3D KDE Isosurface of Multivariate Returns (Kernel Type: Gaussian)",
    )

    fig.show()

def plot_multivariate_gaussian(
    returns_df: pd.DataFrame,
    assets: Tuple[str, str, str],
) -> None:
    if len(assets) != 3:
        raise ValueError("Must provide exactly 3 assets to plot 3D KDE.")
    
    Y = returns_df[assets].dropna().to_numpy()       # (n, 3)

    mu = Y.mean(axis=0)
    Sigma = np.cov(Y, rowvar=False)

    rv = multivariate_normal(mean=mu, cov=Sigma)

    # grid (use percentile range to avoid extreme outliers blowing up the view)
    n = 60
    lo = np.quantile(Y, 0.01, axis=0)
    hi = np.quantile(Y, 0.99, axis=0)

    g1 = np.linspace(lo[0], hi[0], n)
    g2 = np.linspace(lo[1], hi[1], n)
    g3 = np.linspace(lo[2], hi[2], n)

    G1, G2, G3 = np.meshgrid(g1, g2, g3, indexing="ij")
    pos = np.stack([G1, G2, G3], axis=-1)  # (n,n,n,3)
    D = rv.pdf(pos)

    thr = np.quantile(D, 0.95)

    fig = go.Figure(
        data=go.Isosurface(
            x=G1.ravel(), y=G2.ravel(), z=G3.ravel(),
            value=D.ravel(),
            isomin=thr, isomax=D.max(),
            surface_count=2,
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
    )

    fig.update_layout(
        scene=dict(xaxis_title=assets[0], yaxis_title=assets[1], zaxis_title=assets[2]),
        title="3D Multivariate Normal Isosurface",
    )
    fig.show()

def plot_multivariate_returns(
    returns_df: pd.DataFrame,
    assets: Tuple[str, str, str],
    type: str
) -> None:
    if len(assets) != 3:
        raise ValueError("Must provide exactly 3 assets to plot 3D KDE.")
    
    if type == "KDE":
        print("Plotting 3D Guassian KDE isosurface of multivariate returns...")
        plot_3d_kde(returns_df, assets)
    
    elif type == "Gaussian":
        print("Plotting Multivariate Normal (Gaussian) isosurface of multivariate returns...")
        plot_multivariate_gaussian(returns_df, assets)
        