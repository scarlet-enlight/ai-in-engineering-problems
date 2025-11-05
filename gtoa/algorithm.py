from typing import Callable, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class GTOA:
    def __init__(self, func: Callable[[np.ndarray], float], dim: int,
                 bounds: Tuple[float, float], population_size: int = 40,
                 Tmax: int = 5000, seed: int = 42):
        self.func = func
        self.D = dim
        lb, ub = bounds
        self.lb = np.full(self.D, lb) if np.isscalar(lb) else np.asarray(lb, dtype=float)
        self.ub = np.full(self.D, ub) if np.isscalar(ub) else np.asarray(ub, dtype=float)
        self.N = population_size
        if self.N % 2 != 0:
            raise ValueError("Population size must be even (paper divides population into equal halves).")
        self.Tmax = Tmax
        self.rng = np.random.default_rng(seed)
        # state
        self.X = np.zeros((self.N, self.D))
        self.f = np.full(self.N, np.inf)
        self.Tcurrent = 0
        self.G = None
        self.G_f = np.inf
        # history
        self.history_T = []
        self.history_best = []
        self.history_iter = []

    def initialize_population(self):
        # Eq.(11)
        self.X = self.lb + (self.ub - self.lb) * self.rng.random((self.N, self.D))
        # Step 2: evaluate all and Tcurrent += N (Eq.12)
        for i in range(self.N):
            self.f[i] = float(self.func(self.X[i]))
        self.Tcurrent += self.N
        best_idx = int(np.argmin(self.f))
        self.G = self.X[best_idx].copy()
        self.G_f = float(self.f[best_idx])

    def teacher_allocation(self):
        # Eq.(9): pick first three best and form teacher = x_first if f(x_first) <= f(mean_three) else mean_three
        order = np.argsort(self.f)
        x_first = self.X[order[0]]
        x_second = self.X[order[1]]
        x_third = self.X[order[2]]
        mean_three = (x_first + x_second + x_third) / 3.0
        # compute f for first and mean_three (we compute them; block counting is handled separately)
        f_first = float(self.func(x_first))
        f_mean = float(self.func(mean_three))
        return x_first.copy() if f_first <= f_mean else mean_three.copy()

    def ability_grouping(self):
        order = np.argsort(self.f)
        half = self.N // 2
        good_idx = order[:half].tolist()
        bad_idx = order[half:].tolist()
        return good_idx, bad_idx

    def teacher_phase_good(self, T, good_idx):
        if len(good_idx) == 0:
            return
        M = np.mean(self.X[good_idx], axis=0)
        for idx in good_idx:
            a = self.rng.random()
            b = self.rng.random()
            c = self.rng.random()
            F = int(self.rng.choice([1,2]))
            x_i = self.X[idx]
            # Eq.(2)
            x_new = x_i + a * (T - F * (b * M + c * x_i))
            x_new = np.clip(x_new, self.lb, self.ub)
            f_new = float(self.func(x_new))
            # acceptance Eq.(6)
            if f_new < self.f[idx]:
                self.X[idx] = x_new
                self.f[idx] = f_new
                if f_new < self.G_f:
                    self.G_f = f_new
                    self.G = x_new.copy()

    def teacher_phase_bad(self, T, bad_idx):
        if len(bad_idx) == 0:
            return
        for idx in bad_idx:
            d = self.rng.random()
            x_i = self.X[idx]
            # Eq.(5)
            x_new = x_i + 2.0 * d * (T - x_i)
            x_new = np.clip(x_new, self.lb, self.ub)
            f_new = float(self.func(x_new))

            # 6?
            if f_new < self.f[idx]:
                self.X[idx] = x_new
                self.f[idx] = f_new
                if f_new < self.G_f:
                    self.G_f = f_new
                    self.G = x_new.copy()

    def student_phase(self, group_idx, X_before_teacher):
        # group_idx: list of indices to do student-phase for
        if len(group_idx) == 0:
            return
        for idx in group_idx:
            peers = list(range(self.N))
            peers.remove(idx)
            j = int(self.rng.choice(peers))
            e = self.rng.random()
            g = self.rng.random()
            x_teacher_i = self.X[idx].copy()      # after teacher-phase update
            x_teacher_j = self.X[j].copy()
            x_old_i = X_before_teacher[idx].copy()  # IMPORTANT: value at start of teacher-phase (x_i^t)
            # Eq.(7): choose structure based on current fitness comparison
            if self.f[idx] < self.f[j]:
                x_student = x_teacher_i + e * (x_teacher_i - x_teacher_j) + g * (x_teacher_i - x_old_i)
            else:
                x_student = x_teacher_i + e * (x_teacher_j - x_teacher_i) + g * (x_teacher_i - x_old_i)
            x_student = np.clip(x_student, self.lb, self.ub)
            f_student = float(self.func(x_student))
            f_teacher_i = float(self.f[idx])
            # Eq.(8): choose better
            if f_student < f_teacher_i:
                self.X[idx] = x_student
                self.f[idx] = f_student
                if f_student < self.G_f:
                    self.G_f = f_student
                    self.G = x_student.copy()

    def optimize(self, verbose: bool = False, iter_limit: int = 1000):
        # Steps 1 & 2
        self.initialize_population()
        iter_count = 0
        while self.Tcurrent <= self.Tmax:
            iter_count += 1
            # Step 4: teacher allocation (Eq.9)
            T = self.teacher_allocation()
            # Step 5: ability grouping
            good_idx, bad_idx = self.ability_grouping()
            # Save X before teacher-phase for student-phase self-learning (x_i^t)
            X_before_teacher = self.X.copy()
            # Step 6.1: teacher phase for good and student-phase for good (Eq.2, Eq.7-8)
            self.teacher_phase_good(T, good_idx)
            self.student_phase(good_idx, X_before_teacher)
            # Step 6.2: teacher phase for bad and student-phase for bad (Eq.5, Eq.7-8)
            self.teacher_phase_bad(T, bad_idx)
            self.student_phase(bad_idx, X_before_teacher)
            # Step 8: re-evaluate population (Eq.13 block accounting)
            for i in range(self.N):
                self.f[i] = float(self.func(self.X[i]))
            self.Tcurrent += (2 * self.N + 1)
            # update global best
            best_idx = int(np.argmin(self.f))
            if self.f[best_idx] < self.G_f:
                self.G_f = float(self.f[best_idx])
                self.G = self.X[best_idx].copy()
            # record history
            self.history_T.append(self.Tcurrent)
            self.history_best.append(self.G_f)
            self.history_iter.append(iter_count)
            if verbose and (iter_count % 10 == 0):
                print(f"Iter {iter_count}, Tcurrent={self.Tcurrent}, best={self.G_f:.6g}")
            if iter_count >= iter_limit:
                break
        return self.G.copy(), float(self.G_f), int(self.Tcurrent), self.history_iter, self.history_T, self.history_best