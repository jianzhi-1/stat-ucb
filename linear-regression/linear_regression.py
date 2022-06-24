import numpy as np
import pandas as pd

# Generates linear regression data of n points in [low, high]
# characterized by Y = beta0 + beta1*X + e
# where e ~ N(0, std)
def generate_linear_regression_data(n, low, high, beta0, beta1, std):
    x = np.random.uniform(low, high, size=n)
    eps = np.random.normal(0, std, size=n)
    y = beta1*x + beta0 + eps
    return np.column_stack((x, y))

# Returns residual sum of squares
def rss(y, yhat):
    return np.sum((y - yhat)**2)

# Returns (beta0hat, beta1hat) of least square regression
def least_square_regression(x, y):
    n = x.size
    xbar = np.sum(x)/n
    ybar = np.sum(y)/n
    beta1hat = np.dot(x - xbar, y - ybar) / np.sum((x - xbar)**2)
    return (ybar - beta1hat*xbar, beta1hat)

# Returns (beta0hat, beta1hat) of least square regression
def least_square_regression_data(data):
    return least_square_regression(data[:, 0], data[:, 1])

if __name__ == "__main__":
    d = generate_linear_regression_data(100, 0, 100, 4, 1, 1)
    print(least_square_regression_data(d))

