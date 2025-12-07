"""
Base class for all optimization algorithms
"""

from abc import ABC, abstractmethod
from typing import Callable, Tuple
import numpy as np


class BaseOptimizer(ABC):
    """Base class for all optimization algorithms"""
    
    def __init__(self, func: Callable[[np.ndarray], float], dim: int,
                 bounds: Tuple[float, float], population_size: int = 50,
                 max_iterations: int = None, seed: int = None):
        self.func = func
        self.D = dim
        lb, ub = bounds
        self.lb = np.full(self.D, lb) if np.isscalar(lb) else np.asarray(lb, dtype=float)
        self.ub = np.full(self.D, ub) if np.isscalar(ub) else np.asarray(ub, dtype=float)
        self.N = population_size
        self.max_iter = max_iterations if max_iterations is not None else (100 * self.D)
        self.rng = np.random.default_rng(seed)
        
        # State variables
        self.X = np.zeros((self.N, self.D))
        self.f = np.full(self.N, np.inf)
        self.best_position = None
        self.best_fitness = np.inf
        self.current_iter = 0
        self.total_evals = 0
        
        # History
        self.history_iter = []
        self.history_evals = []
        self.history_best = []
    
    def initialize_population(self):
        """Initialize population uniformly within bounds"""
        self.X = self.lb + (self.ub - self.lb) * self.rng.random((self.N, self.D))
        self.f = np.array([self.func(x) for x in self.X])
        self.total_evals += self.N
        
        best_idx = int(np.argmin(self.f))
        self.best_position = self.X[best_idx].copy()
        self.best_fitness = float(self.f[best_idx])
    
    def clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Clip solution to bounds"""
        return np.clip(x, self.lb, self.ub)
    
    def update_best(self):
        """Update global best if better solution found"""
        best_idx = int(np.argmin(self.f))
        if self.f[best_idx] < self.best_fitness:
            self.best_fitness = float(self.f[best_idx])
            self.best_position = self.X[best_idx].copy()
    
    def record_history(self):
        """Record current state in history"""
        self.history_iter.append(self.current_iter)
        self.history_evals.append(self.total_evals)
        self.history_best.append(self.best_fitness)
    
    @abstractmethod
    def iterate(self):
        """Perform one iteration of the algorithm"""
        pass
    
    def optimize(self, verbose: bool = False):
        """Main optimization loop"""
        self.initialize_population()
        self.record_history()
        
        for self.current_iter in range(1, self.max_iter + 1):
            self.iterate()
            self.update_best()
            self.record_history()
            
            if verbose and (self.current_iter % 50 == 0):
                print(f"Iter {self.current_iter}, best={self.best_fitness:.6e}")
        
        return (
            self.best_position.copy(),
            float(self.best_fitness),
            int(self.total_evals),
            self.history_iter,
            self.history_evals,
            self.history_best
        )