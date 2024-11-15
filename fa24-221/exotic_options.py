def C_perpetual(S, K, r, q, sigma):
    """
    Returns the value of a perpetual call.
    """
    import numpy as np
    alpha = (-(r - q - 0.5*sigma**2) + np.sqrt((r - q - 0.5*sigma**2)**2 + 2*r*sigma**2))/(sigma**2)
    H = alpha*K/(alpha - 1.)
    return K/(alpha - 1)*((alpha - 1.)/alpha*(S/K))**alpha if S < H else S - K

def C_binary_cash(H, S, K, r, t, sigma, div):
    """
    Returns the value of a binary cash (European) option
    i.e. payoff = H*1{S_T >= K}
    t: time to maturity
    """
    from scipy.stats import norm
    import numpy as np
    return H*np.exp(-r*t)*norm.cdf((np.log(S/K)+(r-div-sigma**2/2)*t)/(sigma*np.sqrt(t)))
