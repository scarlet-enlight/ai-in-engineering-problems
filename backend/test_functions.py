import numpy as np


# Benchmark functions
def sphere_fn(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * x))


def rastrigin_fn(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    A = 10.0
    n = x.size
    return float(A * n + np.sum(x*x - A * np.cos(2*np.pi*x)))


def rosenbrock_fn(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    # classic Rosenbrock sum_{i=1..n-1} [100*(x_{i+1}-x_i^2)^2 + (x_i-1)^2]
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1.0)**2))


def beale_fn(x: np.ndarray) -> float:
    # defined for 2D only
    x1, x2 = float(x[0]), float(x[1])
    return float((1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*(x2**2))**2 + (2.625 - x1 + x1*(x2**3))**2)


def bukin_n6_fn(x: np.ndarray) -> float:
    # Bukin N.6 defined for x in [-15,-5], y in [-3,3]
    x1, x2 = float(x[0]), float(x[1])
    return float(100.0*np.sqrt(abs(x2 - 0.01*x1**2)) + 0.01*abs(x1 + 10))


# Mapping: name -> (function, list of dims, domain tuple or per-dim)
TEST_FUNCTIONS = {
    "Sphere": {
        "fn": sphere_fn,
        "dims": [2, 5, 10, 20],
        "bounds": (-100.0, 100.0)
    },
    "Rastrigin": {
        "fn": rastrigin_fn,
        "dims": [2, 5, 10],
        "bounds": (-5.12, 5.12)
    },
    "Rosenbrock": {
        "fn": rosenbrock_fn,
        "dims": [2, 5, 10],
        "bounds": (-2.048, 2.048)
    },
    "Beale": {
        "fn": beale_fn,
        "dims": [2],
        "bounds": (-4.5, 4.5)
    },
    "BukinN6": {
        "fn": bukin_n6_fn,
        "dims": [2],
        # Bukin N.6 has different bounds per coordinate: x in [-15,-5], y in [-3,3]
        "bounds": (np.array([-15.0, -3.0]), np.array([-5.0, 3.0]))
    }
}
