"""
Group Teaching Optimization Algorithm (GTOA) implementation
"""

from typing import Callable, Tuple
import numpy as np


class GTOA:
    def __init__(self, func: Callable[[np.ndarray], float], dim: int,
                 bounds: Tuple[float, float], population_size: int = 50,
                 Tmax: int = None, seed: int = None):
        self.func = func
        self.D = dim
        lb, ub = bounds
        self.lb = np.full(self.D, lb) if np.isscalar(lb) else np.asarray(lb, dtype=float)
        self.ub = np.full(self.D, ub) if np.isscalar(ub) else np.asarray(ub, dtype=float)
        self.N = population_size
        if self.N % 2 != 0:
            raise ValueError("Population size must be even (paper divides population into equal halves).")
        self.Tmax = (5000 * self.D) if (Tmax is None) else Tmax
        self.rng = np.random.default_rng(seed)

        # State
        self.X = np.zeros((self.N, self.D))
        self.f = np.full(self.N, np.inf)
        self.Tcurrent = 0
        self.G = None
        self.G_f = np.inf

        # History
        self.history_T = []
        self.history_best = []
        self.history_iter = []

    def initialize_population(self):
        """Initialize population and evaluate fitness"""
        self.X = self.lb + (self.ub - self.lb) * self.rng.random((self.N, self.D))
        self.f = np.array([self.func(x) for x in self.X])
        self.Tcurrent += self.N
        best_idx = int(np.argmin(self.f))
        self.G = self.X[best_idx].copy()
        self.G_f = float(self.f[best_idx])

    def teacher_allocation(self) -> np.ndarray:
        """Select teacher based on Eq.9"""
        order = np.argsort(self.f)
        x_first = self.X[order[0]]
        x_second = self.X[order[1]]
        x_third = self.X[order[2]]
        mean_three = (x_first + x_second + x_third) / 3.0
        f_mean = self.func(mean_three)
        
        return x_first.copy() if self.f[order[0]] <= f_mean else mean_three.copy()

    def ability_grouping(self):
        """Divide population into good and bad halves"""
        order = np.argsort(self.f)
        half = self.N // 2
        return order[:half].tolist(), order[half:].tolist()

    def update_if_better(self, idx: int, x_new: np.ndarray, f_new: float):
        """Update individual and global best if new solution is better"""
        if f_new < self.f[idx]:
            self.X[idx] = x_new
            self.f[idx] = f_new
            if f_new < self.G_f:
                self.G_f = f_new
                self.G = x_new.copy()

    def teacher_phase_good(self, T_teacher: np.ndarray, good_idx):
        """Teacher phase for good students (Eq.2)"""
        if not good_idx:
            return
        M = np.mean(self.X[good_idx], axis=0)
        for idx in good_idx:
            a, b, c = self.rng.random(3)
            F = int(self.rng.choice([1, 2]))
            x_new = self.X[idx] + a * (T_teacher - F * (b * M + c * self.X[idx]))
            x_new = np.clip(x_new, self.lb, self.ub)
            self.update_if_better(idx, x_new, self.func(x_new))

    def teacher_phase_bad(self, T_teacher: np.ndarray, bad_idx):
        """Teacher phase for bad students (Eq.5)"""
        if not bad_idx:
            return
        for idx in bad_idx:
            d = self.rng.random()
            x_new = self.X[idx] + 2.0 * d * (T_teacher - self.X[idx])
            x_new = np.clip(x_new, self.lb, self.ub)
            self.update_if_better(idx, x_new, self.func(x_new))

    def student_phase(self, group_idx, X_before_teacher):
        """Student phase (Eq.7-8)"""
        if not group_idx:
            return
        for idx in group_idx:
            peers = [i for i in range(self.N) if i != idx]
            j = int(self.rng.choice(peers))
            e, g = self.rng.random(2)
            
            x_teacher_i = self.X[idx].copy()
            x_teacher_j = self.X[j].copy()
            x_old_i = X_before_teacher[idx].copy()
            
            if self.f[idx] < self.f[j]:
                x_student = x_teacher_i + e * (x_teacher_i - x_teacher_j) + g * (x_teacher_i - x_old_i)
            else:
                x_student = x_teacher_i + e * (x_teacher_j - x_teacher_i) + g * (x_teacher_i - x_old_i)
            
            x_student = np.clip(x_student, self.lb, self.ub)
            self.update_if_better(idx, x_student, self.func(x_student))

    def optimize(self, verbose: bool = False, iter_limit: int = 1_000_000):
        """Main optimization loop"""
        self.initialize_population()
        iter_count = 0
        
        while self.Tcurrent <= self.Tmax and iter_count < iter_limit:
            iter_count += 1
            T_teacher = self.teacher_allocation()
            good_idx, bad_idx = self.ability_grouping()
            X_before_teacher = self.X.copy()

            self.teacher_phase_good(T_teacher, good_idx)
            self.student_phase(good_idx, X_before_teacher)
            self.teacher_phase_bad(T_teacher, bad_idx)
            self.student_phase(bad_idx, X_before_teacher)
            
            self.Tcurrent += (2 * self.N + 1)

            # Update global best
            best_idx = int(np.argmin(self.f))
            if self.f[best_idx] < self.G_f:
                self.G_f = float(self.f[best_idx])
                self.G = self.X[best_idx].copy()
            
            # Record history
            self.history_T.append(self.Tcurrent)
            self.history_best.append(self.G_f)
            self.history_iter.append(iter_count)
            
            if verbose and (iter_count % 50 == 0):
                print(f"Iter {iter_count}, Tcurrent={self.Tcurrent}, best={self.G_f:.6g}")
        
        return self.G.copy(), float(self.G_f), int(self.Tcurrent), self.history_iter, self.history_T, self.history_best
