def C_perpetual(S, K, r, q, sigma):
    """
    Returns the value of a perpetual call.
    """
    alpha = (-(r - q - 0.5*sigma**2) + np.sqrt((r - q - 0.5*sigma**2)**2 + 2*r*sigma**2))/(sigma**2)
    H = alpha*K/(alpha - 1.)
    return K/(alpha - 1)*((alpha - 1.)/alpha*(S/K))**alpha if S < H else S - K
