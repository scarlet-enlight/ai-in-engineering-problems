import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from gtoa import GTOA


# Parameters (specified in the acrticle)
D = 10                    # Dimension
POP_SIZE = 50             # N = 50 (in the article)
RUNS = 10                 # Number of independent runs (the article uses 30)
TMAX = 5000 * D           # Tmax = 5000 * D for unimodal

# Other parameters
EPS_LOG = 1e-800         # Clipping before log10 (to avoid receiving -inf)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Absolute path to the directory where this script is located
SAVE_DIR = os.path.join(BASE_DIR, "gtoa_results") # Directory for saving csv and plot


# Draw a graph similar to those in the article.
def plot_convergence(T_axis, log10_mean, save_dir=SAVE_DIR,
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
    csv_path = os.path.join(save_dir, "gtoa_experiment_results.csv")
    df.to_csv(csv_path, index=False)

    return T_axis, mean_best, std_best, log10_mean, csv_path
