import numpy as np
from scipy.stats import norm

def c(S, t, sigma, K=950, T=0.25, r=0.04, q=0.02992227099):
    d1 = (np.log(S/K) + (r - q + sigma*sigma/2.)*(T - t))/(sigma*np.sqrt(T - t))
    d2 = (np.log(S/K) + (r - q - sigma*sigma/2.)*(T - t))/(sigma*np.sqrt(T - t))
    return S*np.exp(-q*(T - t))*norm.cdf(d1) - K*np.exp(-r*(T - t))*norm.cdf(d2)

def implied_vol(f, P, eps=1e-6):
    mini = 0.000000
    maxi = 3.000000
    while mini + eps < maxi:
        midi = (mini + maxi)/2.
        if f(midi) < P:
            mini = midi
        else:
            maxi = midi
    return mini

implied_vol(lambda vol: c(1000, 0, vol), 78)
