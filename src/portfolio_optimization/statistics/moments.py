import numpy as np
import yfinance as yf

def coskewness(X):
    T, n = X.shape
    M3 = np.empty((n, 0))
    mu = np.mean(X, axis=0).reshape(n, 1)
    x = X - np.repeat(mu.T, T, axis=0)
    ones = np.ones((1, n))
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

if __name__ == "__main__":
    start = '2019-01-01'
    end = '2023-12-30'
    assets = ['APA', 'BA', 'BAX', 'BMY', 'CMCSA',
          'CNP', 'CPB', 'DE', 'HPQ', 'JCI']
    
    prices = yf.download(assets, start=start, end=end, auto_adjust=False)
    prices = prices.loc[:,('Adj Close', slice(None))]
    prices.columns = assets
    returns = prices[assets].pct_change().dropna()

    R = returns.to_numpy()

    M_3 = coskewness(R)
    M_4 = cokurtosis(R)    
