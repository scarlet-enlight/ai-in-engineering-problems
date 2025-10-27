# Finalized GTOA implementation (exactly per Zhang & Jin 2020 pseudocode) + benchmark tests

from typing import Callable, Tuple, Optional
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

class GTOA_Zhang2020_Final:
    """
    Final version matching Zhang & Jin (2020) pseudocode and flowchart precisely.
    - Uses X_before_teacher for student-phase's self-learning term (Eq.7)
    - Block accounting for function evaluations: Tcurrent += N (init), then += 2N+1 per iteration.
    - Teacher allocation per Eq.(9): compare f(first) and f(mean_three)
    """
    def __init__(self, func: Callable[[np.ndarray], float], dim: int,
                 bounds: Tuple[float, float], population_size: int = 40,
                 Tmax: int = 5000, seed: int = 42):
        #Step 1
        self.Tmax = Tmax
        self.Tcurrent = 0

        self.N = population_size
        if self.N % 2 != 0:
            raise ValueError("Population size must be even (paper divides population into equal halves).")

        lb, ub = bounds
        self.lb = np.full(self.D, lb) if np.isscalar(lb) else np.asarray(lb, dtype=float)
        self.ub = np.full(self.D, ub) if np.isscalar(ub) else np.asarray(ub, dtype=float)

        self.D = dim
        self.func = func
        self.X = np.zeros((self.N, self.D))

        #Step 2
        self.f = np.full(self.N, np.inf)
        self.G = None
        self.G_f = np.inf

        self.rng = np.random.default_rng(seed)
        self.k = self.rng.random((self.N, self.D))

        # History
        self.history_T = []
        self.history_best = []
        self.history_iter = []

    def initialize_population(self):
        # Eq.(11)
        self.X = self.lb + (self.ub - self.lb) * k

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


# ----------------------
# Benchmark functions
# ----------------------
def sphere(x): return float(np.sum(x**2))
def rastrigin(x):
    A = 10.0
    return float(A * x.size + np.sum(x**2 - A * np.cos(2 * math.pi * x)))
def rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))
def ackley(x):
    a = 20; b = 0.2; c = 2 * math.pi
    n = x.size
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    return float(-a * np.exp(-b * math.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + math.e)

# ----------------------
# Run tests
# ----------------------
def run_tests_and_plot(dim=10, N=40, Tmax=5000, seed=12345, verbose=False):
    benchmarks = {
        "Sphere": (sphere, (-5.12, 5.12)),
        "Rastrigin": (rastrigin, (-5.12, 5.12)),
        "Rosenbrock": (rosenbrock, (-2.0, 2.0)),
        "Ackley": (ackley, (-32.768, 32.768))
    }
    results = {}
    plt.rcParams.update({'figure.figsize': (10,6)})
    for name, (func, bounds) in benchmarks.items():
        print(f"\nRunning GTOA final on {name} (D={dim}, N={N}, Tmax={Tmax})")
        opt = GTOA_Zhang2020_Final(func, dim, bounds, population_size=N, Tmax=Tmax, seed=seed)
        best_x, best_val, Tused, iters, T_hist, best_hist = opt.optimize(verbose=verbose)
        results[name] = {
            "best_value": best_val,
            "best_x": best_x,
            "Tused": Tused,
            "iterations": len(iters),
            "T_hist": T_hist,
            "best_hist": best_hist
        }
        print(f"Done {name}: best={best_val:.6g}, Tused={Tused}, iterations={len(iters)}")
        # plot
        plt.figure()
        plt.plot(T_hist, best_hist, linewidth=2)
        plt.xlabel("Tcurrent (approx function evals)")
        plt.ylabel("Best fitness so far")
        plt.title(f"GTOA (final) progress on {name}")
        plt.grid(True)
        plt.show()
    # summary dataframe
    df = pd.DataFrame([{"Function": k, "BestValue": v["best_value"], "Tused": v["Tused"], "Iterations": v["iterations"]} for k,v in results.items()])
    display_df = df[["Function","BestValue","Tused","Iterations"]]
    print("\nSummary:\n", display_df.to_string(index=False))
    return results

# Execute tests
_final_results = run_tests_and_plot(dim=10, N=40, Tmax=5000, seed=12345, verbose=False)
