import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from gtoa import GTOA
from test_functions import TEST_FUNCTIONS

def run_single_experiment(func, bounds, dim, N, I, Tmax, seed, verbose=False):
    opt = GTOA(func, dim, bounds, population_size=N, Tmax=Tmax, seed=seed)
    best_x, best_val, Tused, iters, T_hist, best_hist = opt.optimize(verbose=verbose, iter_limit=I)
    return {
        "best_x": best_x,
        "best_val": best_val,
        "T_hist": T_hist,
        "best_hist": best_hist
    }

def compute_std(values):
    return float(np.std(values))

def compute_cv_percent(values):
    mu = np.mean(values)
    sigma = np.std(values)
    if abs(mu) < 1e-8:
        return float("nan")
    return float((sigma / mu) * 100.0)

def run_experiments(N_values=[10,20,40,80], I_values=[5,10,20,40,60,80],
                    Tmax=5000, n_repeats=10, seed=12345, verbose=False):
    benchmarks = {
        #"Sphere": (sphere, (-5.12, 5.12), 2),
        #"Rastrigin": (rastrigin, (-5.12, 5.12), 2),
        "Rosenbrock": (rosenbrock, (-2.0, 2.0), 5),
        #"Beale": (beale, (-4.5, 4.5), 2),
        #"Bukin": (bukin, (np.array([-15.0, -3.0]), np.array([-5.0, 3.0])), 2)
    }
    all_results = []
    for name, (func, bounds, func_dim) in benchmarks.items():
        print(f"\n=== Funkcja testowa: {name} ===")
        for N in N_values:
            for I in I_values:
                vals = []
                sols = []
                for r in range(n_repeats):
                    res = run_single_experiment(func, bounds, func_dim, N, I, Tmax, seed + r, verbose)
                    vals.append(res["best_val"])
                    sols.append(res["best_x"])
                vals = np.array(vals)
                sols = np.array(sols)
                best_idx = np.argmin(vals)
                worst_idx = np.argmax(vals)
                best_val = vals[best_idx]
                worst_val = vals[worst_idx]
                best_solution = sols[best_idx]
                worst_solution = sols[worst_idx]

                coord_std = np.std(sols, axis=0).tolist()
                coord_cv = [compute_cv_percent(sols[:, j]) for j in range(func_dim)]
                coord_use_std = [bool(np.abs(np.mean(sols[:, j])) < 1e-8) for j in range(func_dim)]

                val_std = float(np.std(vals))
                val_cv = compute_cv_percent(vals)
                print(f"N={N}, I={I} → minimum={best_val:.4e}, najgorsze={worst_val:.4e}, val_cv={val_cv}, val_std={val_std}")
                value_use_std = bool(np.abs(np.mean(vals)) < 1e-8)

                all_results.append({
                    "Algorytm": "GTOA",
                    "Funkcja testowa": name,
                    "Liczba szukanych parametrów": func_dim,
                    "Liczba iteracji": I,
                    "Rozmiar populacji": N,
                    "Znalezione minimum (wektor)": best_solution.tolist(),
                    "Odchylenie std. parametrów (wektor)": coord_std,
                    "Współczynnik zmienności parametrów (wektor, %)": [float(x) for x in coord_cv],
                    "Użyto std zamiast CV dla współrzędnych (bool per coord)": coord_use_std,
                    "Wartość funkcji celu (min)": float(best_val),
                    "Odchylenie std. wartości funkcji celu": val_std,
                    "Współczynnik zmienności wartości funkcji celu (%)": val_cv,
                    "Użyto std zamiast CV dla wartości funkcji (bool)": value_use_std,
                    "Najgorsze rozwiązanie (wartość funkcji)": float(worst_val),
                    "Najgorsze rozwiązanie (wektor)": worst_solution.tolist()
                })
    df = pd.DataFrame(all_results)
    df.to_csv("results_summary.csv", index=False)
    print("\n📊 Wyniki zapisano do results_summary.csv")
    return df

def plot_results(df, function_name):
    plt.figure(figsize=(10,6))
    pivot = df[df["Funkcja testowa"] == function_name].pivot(index="Rozmiar populacji", columns="Liczba iteracji", values="Wartość funkcji celu (min)")
    plt.imshow(pivot, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(label="Najlepsza (minimalna) wartość funkcji celu")
    plt.title(f"{function_name} — minimum dla różnych N i I")
    plt.xlabel("Liczba iteracji (I)")
    plt.ylabel("Rozmiar populacji (N)")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.show()


# Parameters (specified in the acrticle)
D = 10                    # Dimension
POP_SIZE = 50             # N = 50 (in the article)
RUNS = 10                 # Number of independent runs (the article uses 30)
TMAX = 5000 * D           # Tmax = 5000 * D for unimodal
EPS_LOG = 1e-800         # Clipping before log10 (to avoid receiving -inf)
SAVE_DIR = "gtoa_results" # Directory for csv saving csv and plot


# Draw a graph similar to those in the article.
def plot_convergence(T_axis, mean_best, std_best, log10_mean, save_dir=SAVE_DIR,
                     title="Convergence of GTOA"):
    plt.figure(figsize=(8,5))
    
    # Y = log10(mean fitness). Displaying negative values (—200 and etc)
    plt.plot(T_axis, log10_mean, linewidth=2, color='tab:orange', label='GTOA (log10(mean best))')
    plt.ylabel('Mean fitness value / log10')

    plt.xlabel('The number of function evaluations')
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    png_path = os.path.join(save_dir, f"gtoa.png")
    plt.savefig(png_path, dpi=300)
    plt.show()
    return png_path


# Experiment: RUNS of independent runs and averaging
def run_experiment(func, bounds, runs: int = RUNS, pop: int = POP_SIZE, D: int = D, Tmax: int = TMAX,
                   seed_base: int = 0, save_dir: str = SAVE_DIR, verbose: bool = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_hist_T = []
    all_hist_best = []

    for run in range(runs):
        seed = seed_base + run
        g = GTOA(func, dim=D, bounds=bounds, population_size=pop, Tmax=Tmax, seed=seed)
        _, best_val, Tfinal, iters, hist_T, hist_best = g.optimize(verbose=verbose)
        all_hist_T.append(np.array(hist_T))
        all_hist_best.append(np.array(hist_best))
        if verbose:
            print(f"Run {run+1}/{runs}: final best={best_val:.3e}, FE final={Tfinal}")

    # align the lengths (padding with the last value)
    max_len = max(len(h) for h in all_hist_best)
    for i in range(runs):
        if len(all_hist_best[i]) < max_len:
            last_T = all_hist_T[i][-1]
            last_best = all_hist_best[i][-1]
            needed = max_len - len(all_hist_best[i])
            all_hist_T[i] = np.concatenate([all_hist_T[i], np.full(needed, last_T)])
            all_hist_best[i] = np.concatenate([all_hist_best[i], np.full(needed, last_best)])

    T_matrix = np.vstack(all_hist_T)      # shape (runs, steps)
    best_matrix = np.vstack(all_hist_best)

    # Take the X-axis from the first run (they're aligned)
    T_axis = T_matrix[0]
    mean_best = np.mean(best_matrix, axis=0)  # average across runs in the original scale
    std_best = np.std(best_matrix, axis=0)

    # To avoid -inf, clip the minimum value
    log10_mean = np.log10(np.maximum(mean_best, EPS_LOG))

    # Saving data to CSV
    df = pd.DataFrame({
        "FE": T_axis,
        "mean_best": mean_best,
        "std_best": std_best,
        "log10_mean_best": log10_mean
    })
    csv_path = os.path.join(save_dir, "gtoa_sphere_10d_results.csv")
    df.to_csv(csv_path, index=False)

    return T_axis, mean_best, std_best, log10_mean, csv_path

