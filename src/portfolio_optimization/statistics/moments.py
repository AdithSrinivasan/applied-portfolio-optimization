import numpy as np
import yfinance as yf

def coskewness(X): # X is a numpy array
    """
    Vectorized implementation for coskewness calculation...

    Basic mental model:
        for i in assets:
            for j in assets:
                for k in assets:
                    M3[i,j,k] = mean(x[:,i] * x[:,j] * x[:,k])
    """
    T, n = X.shape
    x = X - X.mean(axis=0) # calculates centered (demeaned) returns
    ones = np.ones((1, n)) # row of ones, length: number of assets in returns matrix
    W1 = np.kron(ones, x)
    W2 = np.kron(x, ones)
    M3 = (1/T) * x.T @ (W1 * W2)
    return M3


def cokurtosis(X):
    T, n = X.shape
    M4 = np.empty((n, 0))
    mu = np.mean(X, axis=0).reshape(n, 1)
    x = X - np.repeat(mu.T, T, axis=0)
    ones = np.ones((1, n))
    W1 = np.kron(x, np.kron(ones, ones))
    W2 = np.kron(ones, np.kron(x, ones))
    W3 = np.kron(ones, np.kron(ones, x))
    M4 = (1/T) * x.T @ (W1 * W2 * W3)
    return M4