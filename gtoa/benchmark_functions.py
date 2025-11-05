import numpy as np

def sphere(x): return float(np.sum(x**2))

def rastrigin(x):
    A = 10.0
    return float(A * x.size + np.sum(x**2 - A * np.cos(2 * np.pi * x)))

def rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))

def beale(x):
    x1 = x[0]
    x2 = x[1] 
    term1 = (1.5 - x1 + x1 * x2)**2
    term2 = (2.25 - x1 + x1 * x2**2)**2
    term3 = (2.625 - x1 + x1 * x2**3)**2  
    return float(term1 + term2 + term3)

def bukin(x): 
    x1 = x[0]
    x2 = x[1]
    term1 = 100.0 * np.sqrt(np.abs(x2 - 0.01 * x1**2))
    term2 = 0.01 * np.abs(x1 + 10.0)
    return float(term1 + term2)