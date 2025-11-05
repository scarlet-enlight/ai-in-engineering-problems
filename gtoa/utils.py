import matplotlib.colors as mcolors
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
        "Sphere": (sphere, (-5.12, 5.12), 2),
        "Rastrigin": (rastrigin, (-5.12, 5.12), 2),
        "Rosenbrock": (rosenbrock, (-2.0, 2.0), 2),
        "Beale": (beale, (-4.5, 4.5), 2),
        "Bukin": (bukin, (np.array([-15.0, -3.0]), np.array([-5.0, 3.0])), 2)
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


def plot_line_results(df, function_name, eps_fallback=1e-16):
    subset = df[df["Funkcja testowa"] == function_name]
    plt.figure(figsize=(10,6))

    # выбираем eps: если есть ненулевые положительные значения, берём 1/10 минимального ненулевого
    nonzeros = subset["Wartość funkcji celu (min)"][subset["Wartość funkcji celu (min)"] > 0]
    eps = (nonzeros.min() * 0.1) if len(nonzeros) > 0 else eps_fallback

    star_plotted = False
    # гарантируем порядок итераций
    for N in sorted(subset["Rozmiar populacji"].unique()):
        data_N = subset[subset["Rozmiar populacji"] == N].sort_values("Liczba iteracji")
        x = data_N["Liczba iteracji"].values
        y_orig = data_N["Wartość funkcji celu (min)"].values.astype(float)

        # убираем NaN (они разрывают линию)
        mask_not_nan = ~np.isnan(y_orig)
        x = x[mask_not_nan]
        y_orig = y_orig[mask_not_nan]

        # где были нули — отдельная маска
        zero_mask = (y_orig == 0)
        nonzero_mask = ~zero_mask

        # для линии: заменяем нули на eps, чтобы линия была непрерывна на лог-оси
        y_for_line = y_orig.copy()
        if zero_mask.any():
            y_for_line[zero_mask] = eps

        # если есть ненулевые (после фильтра NaN) — рисуем линию через все точки (с eps вместо 0)
        if len(x) > 0:
            line = plt.plot(x, y_for_line, marker='o', label=f'N={N}')[0]
            color = line.get_color()
        else:
            # нет точек вообще — пропускаем
            continue

        # рисуем звёздочки только там, где было точно 0
        if zero_mask.any():
            plt.scatter(x[zero_mask], np.full(zero_mask.sum(), eps),
                        marker='*', s=120, color=color, edgecolors='k', zorder=10)
            star_plotted = True

    plt.yscale('log')
    plt.xlabel("Liczba iteracji (I)")
    plt.ylabel("Najlepsza wartość funkcji celu (log)")
    plt.title(f"{function_name}: wpływ liczby iteracji i rozmiaru populacji")

    # легенда + пояснение про ★
    handles, labels = plt.gca().get_legend_handles_labels()
    if star_plotted:
        star_handle = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
                                 markeredgecolor='k', markersize=12, linestyle='None')
        handles.append(star_handle)
        labels.append("★ = wynik = 0 (globalne minimum)")
    plt.legend(handles, labels, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_surface_results(df, function_name):
    subset = df[df["Funkcja testowa"] == function_name]
    pivot = subset.pivot(index="Rozmiar populacji", columns="Liczba iteracji", values="Wartość funkcji celu (min)")
    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_title(f"{function_name} — powierzchnia wyników GTOA")
    ax.set_xlabel("Liczba iteracji (I)")
    ax.set_ylabel("Rozmiar populacji (N)")
    ax.set_zlabel("Wartość funkcji celu")
    plt.show()

def plot_results_logscale(df, function_name):
    pivot = df[df["Funkcja testowa"] == function_name].pivot(
        index="Rozmiar populacji",
        columns="Liczba iteracji",
        values="Wartość funkcji celu (min)"
    ).sort_index(ascending=True)

    mask_zeros = (pivot == 0)

    nonzero = pivot[pivot > 0].stack()
    eps = (nonzero.min() * 0.1) if len(nonzero) > 0 else 1e-16
    pivot_safe = pivot.replace(0, eps)

    norm = mcolors.LogNorm(vmin=pivot_safe.min().min(), vmax=pivot_safe.max().max())
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_under('navy')

    plt.figure(figsize=(10,6))
    img = plt.imshow(pivot_safe.values, cmap=cmap, origin='lower', aspect='auto', norm=norm)
    plt.colorbar(img, label="log10(Wartość funkcji celu)")

    for (i, j), val in np.ndenumerate(mask_zeros.values):
        if val:
            plt.text(j, i, "★", color="white", ha="center", va="center", fontsize=14, fontweight='bold')

    plt.title(f"{function_name} — log-skalowana mapa wyników")
    plt.xlabel("Liczba iteracji (I)")
    plt.ylabel("Rozmiar populacji (N)")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.tight_layout()
    plt.show()