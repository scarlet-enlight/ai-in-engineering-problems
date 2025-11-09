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
        # According to the article: if Tmax is not transmitted, set 5000*D for unimodal
        self.Tmax = (5000 * self.D) if (Tmax is None) else Tmax
        self.rng = np.random.default_rng(seed)

        # State
        self.X = np.zeros((self.N, self.D))
        self.f = np.full(self.N, np.inf)
        self.Tcurrent = 0  # number of objective function evaluations (FE)
        self.G = None
        self.G_f = np.inf

        # History
        self.history_T = []
        self.history_best = []
        self.history_iter = []

    def evaluate_population(self):
        """Re-evaluates all individuals in the population"""
        self.f = np.array([self.func(x) for x in self.X])

    # Initialization (Eq.11, Eq.12) — Evaluate the initial population, Tcurrent += N
    def initialize_population(self):
        self.X = self.lb + (self.ub - self.lb) * self.rng.random((self.N, self.D))
        self.evaluate_population()  # Evaluate the initial population
        self.Tcurrent += self.N
        best_idx = int(np.argmin(self.f))
        self.G = self.X[best_idx].copy()
        self.G_f = float(self.f[best_idx])


    # Teacher allocation (Eq.9)
    # Compare f(x_first) and f(mean_three); FE counted later (+2N+1 as in paper)
    def teacher_allocation(self) -> np.ndarray:
        order = np.argsort(self.f)
        x_first = self.X[order[0]]
        x_second = self.X[order[1]]
        x_third = self.X[order[2]]
        mean_three = (x_first + x_second + x_third) / 3.0
        # Use the known self.f[order[0]] and calculate f(mean_three) for comparison
        f_mean = self.func(mean_three)
        # We don't increment Tcurrent here - we consider it part of 2N+1 at the end of the step, as in the article
        if self.f[order[0]] <= f_mean:
            return x_first.copy()
        else:
            return mean_three.copy()

    # Ability grouping (Divide the population into good and bad by fitness)
    def ability_grouping(self):
        order = np.argsort(self.f)
        half = self.N // 2
        good_idx = order[:half].tolist()
        bad_idx = order[half:].tolist()
        return good_idx, bad_idx

    # Teacher phase for good (Eq.2)
    def teacher_phase_good(self, T_teacher: np.ndarray, good_idx):
        if len(good_idx) == 0:
            return
        M = np.mean(self.X[good_idx], axis=0)
        for idx in good_idx:
            a = self.rng.random()
            b = self.rng.random()
            c = self.rng.random()
            F = int(self.rng.choice([1, 2]))
            x_i = self.X[idx]
            x_new = x_i + a * (T_teacher - F * (b * M + c * x_i))
            x_new = np.clip(x_new, self.lb, self.ub)
            f_new = self.func(x_new)
            if f_new < self.f[idx]:
                self.X[idx] = x_new
                self.f[idx] = f_new
                if f_new < self.G_f:
                    self.G_f = f_new
                    self.G = x_new.copy()

    # Teacher phase for bad (Eq.5)
    def teacher_phase_bad(self, T_teacher: np.ndarray, bad_idx):
        if len(bad_idx) == 0:
            return
        for idx in bad_idx:
            d = self.rng.random()
            x_i = self.X[idx]
            x_new = x_i + 2.0 * d * (T_teacher - x_i)
            x_new = np.clip(x_new, self.lb, self.ub)
            f_new = self.func(x_new)
            if f_new < self.f[idx]:
                self.X[idx] = x_new
                self.f[idx] = f_new
                if f_new < self.G_f:
                    self.G_f = f_new
                    self.G = x_new.copy()

    # Student phase (Eq.7-8)
    def student_phase(self, group_idx, X_before_teacher):
        if len(group_idx) == 0:
            return
        for idx in group_idx:
            peers = list(range(self.N))
            peers.remove(idx)
            j = int(self.rng.choice(peers))
            e = self.rng.random()
            g = self.rng.random()
            x_teacher_i = self.X[idx].copy()
            x_teacher_j = self.X[j].copy()
            x_old_i = X_before_teacher[idx].copy()
            if self.f[idx] < self.f[j]:
                x_student = x_teacher_i + e * (x_teacher_i - x_teacher_j) + g * (x_teacher_i - x_old_i)
            else:
                x_student = x_teacher_i + e * (x_teacher_j - x_teacher_i) + g * (x_teacher_i - x_old_i)
            x_student = np.clip(x_student, self.lb, self.ub)
            f_student = self.func(x_student)
            f_teacher_i = float(self.f[idx])
            if f_student < f_teacher_i:
                self.X[idx] = x_student
                self.f[idx] = f_student
                if f_student < self.G_f:
                    self.G_f = f_student
                    self.G = x_student.copy()

    # Main loop; accounting for FE per item: +N at initialization, + (2N + 1) after iteration
    def optimize(self, verbose: bool = False, iter_limit: int = 1_000_000):
        self.initialize_population()
        iter_count = 0
        while self.Tcurrent <= self.Tmax:
            iter_count += 1
            T_teacher = self.teacher_allocation()
            good_idx, bad_idx = self.ability_grouping()
            X_before_teacher = self.X.copy()

            # Teacher & student phases
            self.teacher_phase_good(T_teacher, good_idx)
            self.student_phase(good_idx, X_before_teacher)
            self.teacher_phase_bad(T_teacher, bad_idx)
            self.student_phase(bad_idx, X_before_teacher)
            
            
            # According to the article: Tcurrent += 2N + 1 (blockwise)
            self.Tcurrent += (2 * self.N + 1)

            # Update best global
            best_idx = int(np.argmin(self.f))
            if self.f[best_idx] < self.G_f:
                self.G_f = float(self.f[best_idx])
                self.G = self.X[best_idx].copy()
            
            # Recording the history
            self.history_T.append(self.Tcurrent)
            self.history_best.append(self.G_f)
            self.history_iter.append(iter_count)
            if verbose and (iter_count % 50 == 0):
                print(f"Iter {iter_count}, Tcurrent={self.Tcurrent}, best={self.G_f:.6g}")
            if iter_count >= iter_limit:
                break
        return self.G.copy(), float(self.G_f), int(self.Tcurrent), self.history_iter, self.history_T, self.history_best
