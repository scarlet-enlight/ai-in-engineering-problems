"""
gtoa_testbench.py

Launch benchmark tests for GTOA algorithm
"""

import time
import numpy as np
import os
import pandas as pd

from test_functions import TEST_FUNCTIONS


# Setup parameters
ITER_STOP_MODE = 'iter'  # 'iter' or 'FE' mode
n_runs = 10 # Number of runs (repetitions)
N_values = [10, 20, 40, 80] # Population values
I_values = [5, 10, 20, 40, 60, 80] # Iteration values

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Absolute path to the directory where this script is located
SAVE_DIR = os.path.join(BASE_DIR, "gtoa_results") # Directory for saving summary results


# Main benchmark function
def run_benchmark(GTOA_class):
    summary_rows = []  # Summary table for each function 
    total_start = time.time()

    for fname, meta in TEST_FUNCTIONS.items():
        fn = meta["fn"]
        dims_list = meta["dims"]
        bounds = meta["bounds"]

        for D in dims_list:
            # build per-dim bounds array
            if isinstance(bounds, tuple) and isinstance(bounds[0], np.ndarray):
                lb = bounds[0].copy()
                ub = bounds[1].copy()
            else:
                lb = np.full(D, bounds[0], dtype=float)
                ub = np.full(D, bounds[1], dtype=float)

            for N in N_values:
                for I in I_values:
                    print(f"[{fname} D={D}] N={N}, I={I} ...")
                    runs_data = []
                    for run_idx in range(n_runs):
                        seed = 1000*run_idx + 42
                        # For ITER_STOP_MODE == 'iter', pass iter_limit=I
                        # For ITER_STOP_MODE == 'FE', translate iterations I -> the corresponding T_max (approximately)
                        Tmax_for_run = None
                        iter_limit = None
                        if ITER_STOP_MODE == 'iter':
                            iter_limit = I
                        else:
                            # rough approximation: I iterations => T_increments ~ I*(2*N+1) + N(init)
                            Tmax_for_run = I * (2 * N + 1) + N
                        
                        # bounds passed as tuple (lb_scalar, ub_scalar) or arrays expected by your GTOA
                        if isinstance(bounds, tuple) and isinstance(bounds[0], np.ndarray):
                            bnds = (lb, ub)
                        else:
                            bnds = (float(bounds[0]), float(bounds[1]))
                        
                        # Instantiate GTOA
                        g = GTOA_class(func=fn, dim=D, bounds=bnds, population_size=N,
                                       Tmax=Tmax_for_run, seed=seed)
                        # Run
                        Gbest, best_val, *_ = g.optimize(verbose=False, iter_limit=I)

                        # Save one run
                        solution_vector = np.asarray(Gbest).reshape(-1)
                        runs_data.append({
                            "run": run_idx,
                            "seed": seed,
                            "best_val": float(best_val),
                            "FE_final": int(g.Tcurrent),
                            "iters": len(g.history_iter),
                            "x": solution_vector
                        })

                    # Postprocessing 10 runs
                    # Collect an array of vectors (n_runs, D)
                    Xs = np.vstack([r["x"] for r in runs_data])
                    fvals = np.array([r["best_val"] for r in runs_data], dtype=float)

                    # Best of 10
                    best_idx_overall = int(np.argmin(fvals))
                    best_solution = Xs[best_idx_overall]
                    best_value = float(fvals[best_idx_overall])
                    worst_value = float(np.max(fvals))

                    # For each coordinate calculate the mean, std and coefficient of variation (or std if mean~0)
                    coord_means = np.mean(Xs, axis=0)
                    coord_stds = np.std(Xs, axis=0, ddof=0)
                    coord_cv = []
                    for j in range(Xs.shape[1]):
                        mu = coord_means[j]
                        sigma = coord_stds[j]
                        if abs(mu) < 1e-12:
                            coord_cv.append({"type": "std", "value": float(sigma)})
                        else:
                            coord_cv.append({"type": "cv_percent", "value": float((sigma / mu)*100.0)})

                    # For function value
                    f_std = float(np.std(fvals, ddof=0))

                    # Summary row
                    summary_row = {
                        "Algorytm": g.__class__.__name__,
                        "P1": getattr(g, "Tmax", None),    # Max number of FE
                        "Funkcja testowa": fname,
                        "Liczba parametrów": D,
                        "Liczba iteracji": I,
                        "Rozmiar populacji": N,
                        "Znalezione minimum": "(" + ", ".join([f"{xv:.4f}" for xv in best_solution]) + ")",
                        "Odchylenie std parametrów": "(" + ", ".join([f"{s:.4f}" for s in coord_stds]) + ")",
                        "Wartość funkcji celu": f"{best_value:.3e}",
                        "Wartość funkcji celu (najgorsza)": f"{worst_value:.3e}",
                        "Odchylenie std funkcji celu": f"{f_std:.3f}"
                    }

                    summary_rows.append(summary_row)


    total_end = time.time()
    print("All tests finished. Total time (s):", total_end - total_start)

    # Summary table:
    all_summary_df = pd.DataFrame(summary_rows)
    all_summary_df = all_summary_df.fillna("")

    # Ensure directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    csv_path = os.path.join(SAVE_DIR, "summary_results.csv")
    all_summary_df.to_csv(csv_path, index=False)
    print(f"Saved all_summary to: {csv_path}")
    return all_summary_df


# Run testbench
if __name__ == "__main__":
    from gtoa import GTOA as GTOA_impl

    df = run_benchmark(GTOA_impl)
    print("Done. Summary rows:", len(df))
