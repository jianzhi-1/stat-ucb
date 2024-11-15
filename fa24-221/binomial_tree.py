import numpy as np

class Node():
    def __init__(self, S, t, config, depth, max_depth, parent):
        self.S = S
        self.t = t
        if "dt" not in config: config["dt"] = config["T"]/config["N"]
        if "u" not in config: config["u"] = np.exp(config["sigma"]*np.sqrt(config["dt"]))
        if "d" not in config: config["d"] = np.exp(-config["sigma"]*np.sqrt(config["dt"]))
        if "p" not in config: config["p"] = (np.exp((config["r"] - config["q"])*config["dt"]) - config["d"])/(config["u"] - config["d"])
        self.dt = config["dt"]
        self.u = config["u"]
        self.d = config["d"]
        self.p = config["p"]
        self.K = config["K"]
        self.r = config["r"]
        self.american = config["american"]
        assert type(self.american) == bool, "americanness must be a boolean"
        self.depth = depth
        self.max_depth = max_depth
        self.greeks = dict()
        assert config["ty"] in ["call", "put"], "invalid option type"
        self.ty = config["ty"] # type
        self.v = None # value
        self.uptick = None
        self.downtick = None
        self.parent = parent
        if depth < max_depth:
            if parent is not None and self.parent.uptick is not None and self.parent.uptick.downtick is not None:
                self.uptick = self.parent.uptick.downtick # memoisation
            else:
                self.uptick = Node(S*self.u, t + self.dt, config, depth + 1, max_depth, self)
            self.downtick = Node(S*self.d, t + self.dt, config, depth + 1, max_depth, self)
        self.done = False # not analysed yet
        return
    
    def calc(self):
        if self.analysed(): return # already analysed
        
        if self.depth == self.max_depth:
            if self.ty == "call": 
                self.v = max(self.S - self.K, 0.)
                self.greeks["delta"] = 1. if self.S - self.K >= 0. else 0.
                self.greeks["gamma"] = None
                self.greeks["theta"] = None
            else:
                self.v = max(self.K - self.S, 0.)
                self.greeks["delta"] = -1. if self.K - self.S >= 0. else 0.
                self.greeks["gamma"] = None
                self.greeks["theta"] = None
            return
        
        self.uptick.calc()
        self.downtick.calc()
        
        self.v = np.exp(-self.r*self.dt)*(self.p*self.uptick.v + (1. - self.p)*self.downtick.v)
        if self.american:
            if self.ty == "call":
                self.v = max(self.v, max(self.S - self.K, 0.))
            else:
                self.v = max(self.v, max(self.K - self.S, 0.))
        self.greeks["delta"] = (self.uptick.v - self.downtick.v)/(self.uptick.S - self.downtick.S)
        self.greeks["gamma"] = (self.uptick.greeks["delta"] - self.downtick.greeks["delta"])/(self.uptick.S - self.downtick.S)
        if self.uptick.downtick is not None:
            self.greeks["theta"] = (self.uptick.downtick.v - self.v)/(2.*self.dt)
            
        self.done = True # analysed
        return
    
    def analysed(self):
        return self.done
    
    def get_greeks(self):
        return self.greeks

def build_tree(S0, T, N, K, r, q, sigma, ty, american, center=False, eps=0.001):
    config = {
        # simulation properties
        "T": T,
        "N": N,
        # black scholes properties
        "r": r,
        "q": q,
        "sigma": sigma,
        # option properties
        "K": K,
        "ty": ty,
        "american": True
    }
    root = Node(S0, 0, config, 0, N, None)
    root.calc()
    greeks = root.get_greeks()
    
    if center:
        v_p_sigma, _ = build_tree(S0, T, N, K, r, q, sigma + eps, ty, american, center=False)
        v_m_sigma, _ = build_tree(S0, T, N, K, r, q, sigma - eps, ty, american, center=False)
        greeks["vega"] = (v_p_sigma - v_m_sigma)/(2.*eps)
        
        v_p_r, _ = build_tree(S0, T, N, K, r + eps, q, sigma, ty, american, center=False)
        v_m_r, _ = build_tree(S0, T, N, K, r - eps, q, sigma, ty, american, center=False)
        greeks["rho"] = (v_p_r - v_m_r)/(2.*eps)
    
    return root.v, greeks

def main():
    N = 100 # number of steps
    S0 = 100.
    K = 100.
    T = 1.
    r = 0.06
    q = 0.06
    sigma = 0.35
    ty = "call"
    american = True
    
    v, greeks = build_tree(S0, T, N, K, r, q, sigma, ty, american, center=True)
    
    print(f"Value = {round(v, 4)}")
    print(f"Delta = {round(greeks['delta'], 4)}")
    print(f"Gamma = {round(greeks['gamma'], 4)}")
    print(f"Theta = {round(greeks['theta'], 4)}")
    print(f"Vega = {round(greeks['vega'], 4)}")
    print(f"Rho = {round(greeks['rho'], 4)}")

if __name__ == "__main__":
    main()
