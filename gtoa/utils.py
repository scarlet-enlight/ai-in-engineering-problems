import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gtoa.algorithm import GTOA
from gtoa.benchmark_functions import sphere, rastrigin, rosenbrock, beale, bukin

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
        #"Sphere": (sphere, (-5.12, 5.12), 10),
        #"Rastrigin": (rastrigin, (-5.12, 5.12), 10),
        "Rosenbrock": (rosenbrock, (-2.0, 2.0), 10),
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