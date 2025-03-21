import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

def parse_csv_data(file_path):
    """Parse il file CSV con i dati delle prestazioni delle matrici."""
    data = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def categorize_matrices(data):
    """Categorizza le matrici per dimensione basandosi sul numero di elementi non zero."""
    # Categorie con colori associati
    categories = {
        '< 10K': 'blue',
        '10K-100K': 'orange',
        '100K-500K': 'green',
        '500K-1M': 'red',
        '1M-2.5M': 'purple',
        '≥ 2.5M': 'brown'
    }

    matrix_sizes = {}
    for row in data:
        matrix_name = row['matrix_name']
        nonzeros = int(row['nonzeros'])
        matrix_sizes[matrix_name] = nonzeros

    matrix_categories = {cat: [] for cat in categories}
    for matrix, size in matrix_sizes.items():
        if size < 10_000:
            matrix_categories['< 10K'].append(matrix)
        elif size < 100_000:
            matrix_categories['10K-100K'].append(matrix)
        elif size < 500_000:
            matrix_categories['100K-500K'].append(matrix)
        elif size < 1_000_000:
            matrix_categories['500K-1M'].append(matrix)
        elif size < 2_500_000:
            matrix_categories['1M-2.5M'].append(matrix)
        else:
            matrix_categories['≥ 2.5M'].append(matrix)

    return matrix_categories, categories, matrix_sizes

def print_speedup_by_category(data, matrix_categories, speedup_column):
    """Stampa i valori di speedup dalla colonna specificata, raggruppati per categoria."""
    # Raggruppa i dati per categoria e numero di thread
    category_thread_speedup = {cat: {} for cat in matrix_categories}

    for row in data:
        matrix_name = row['matrix_name']
        num_threads = int(row['num_threads'])
        if row[speedup_column] and row[speedup_column] != '':
            speedup = float(row[speedup_column])
        else:
            continue

        # Trova a quale categoria appartiene questa matrice
        for cat, matrices in matrix_categories.items():
            if matrix_name in matrices:
                if num_threads not in category_thread_speedup[cat]:
                    category_thread_speedup[cat][num_threads] = []
                category_thread_speedup[cat][num_threads].append(speedup)
                break

    # Calcola lo speedup medio per ogni categoria e numero di thread
    all_threads = set()
    for cat_data in category_thread_speedup.values():
        all_threads.update(cat_data.keys())
    thread_list = sorted(all_threads)

    # Stampa l'intestazione
    header = "Category".ljust(15)
    for t in thread_list:
        header += f"{t} threads".ljust(15)
    print(f"\nAverage {speedup_column} by category and thread count:")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Stampa le righe di dati e raccoglie i valori medi
    avg_values = {}
    for cat, thread_data in category_thread_speedup.items():
        row = cat.ljust(15)
        avg_values[cat] = []

        for t in thread_list:
            if t in thread_data and thread_data[t]:
                avg = np.mean(thread_data[t])
                avg_values[cat].append(avg)
                row += f"{avg:.2f}".ljust(15)
            else:
                avg_values[cat].append(float('nan'))
                row += "-".ljust(15)
        print(row)

    print("-" * len(header))

    # Stampa anche i valori grezzi per facilitare il copia-incolla
    print(f"\nRaw values for {speedup_column} (for plotting):")
    for cat in matrix_categories:
        print(f"{cat}:", end=" ")
        print(", ".join([f"{v}" for v in avg_values[cat]]))

    return avg_values, thread_list

def plot_individual_speedup(avg_values_dict, thread_list, categories_with_colors):
    """Crea grafici individuali per ogni metrica di speedup con tutte le categorie."""
    metrics = list(avg_values_dict.keys())
    categories = list(categories_with_colors.keys())

    # Crea un grafico separato per ogni metrica
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        for cat in categories:
            color = categories_with_colors[cat]
            values = avg_values_dict[metric][cat]

            # Filtra i valori NaN
            x_values = []
            y_values = []
            for i, val in enumerate(values):
                if not np.isnan(val):
                    x_values.append(thread_list[i])
                    y_values.append(val)

            if x_values:  # Grafica solo se abbiamo dati
                ax.plot(x_values, y_values, marker='o', label=cat, color=color)

        # Configura il grafico
        ax.set_title(f"{metric} per Thread Count")
        ax.set_xlabel("Number of Threads")
        ax.set_ylabel("Speedup")
        ax.legend()
        ax.grid(True)

        # Aggiungi una linea di riferimento ideale y=x
        max_threads = max(thread_list)
        x_ideal = np.arange(1, max_threads + 1)
        y_ideal = x_ideal
        ax.plot(x_ideal, y_ideal, 'k--', alpha=0.5, label='Ideal Speedup')

        # Salva il grafico
        plt.tight_layout()
        plt.savefig(f"../result/{metric}_by_category.png")
        print(f"Saved plot as {metric}_by_category.png")

def plot_comparison_charts(avg_values_dict, thread_list, categories_with_colors):
    """Crea grafici di confronto tra speedup di implementazioni diverse."""
    categories = list(categories_with_colors.keys())

    # 1. Confronto tra speedup_parallel e speedup_simd
    fig, ax = plt.subplots(figsize=(10, 6))

    for cat in categories:
        color = categories_with_colors[cat]
        parallel_values = np.array(avg_values_dict["speedup_parallel"][cat])
        simd_values = np.array(avg_values_dict["speedup_simd"][cat])

        # Crea liste filtrate per il grafico
        x_parallel = []
        y_parallel = []
        x_simd = []
        y_simd = []

        for i in range(len(thread_list)):
            if i < len(parallel_values) and not np.isnan(parallel_values[i]):
                x_parallel.append(thread_list[i])
                y_parallel.append(parallel_values[i])

            if i < len(simd_values) and not np.isnan(simd_values[i]):
                x_simd.append(thread_list[i])
                y_simd.append(simd_values[i])

        if x_parallel:  # Grafica solo se abbiamo dati
            ax.plot(x_parallel, y_parallel, marker='o', linestyle='-', label=f"{cat} (Parallel)", color=color)

        if x_simd:  # Grafica solo se abbiamo dati
            ax.plot(x_simd, y_simd, marker='s', linestyle='--', label=f"{cat} (SIMD)", color=color, alpha=0.7)

    # Configura il grafico
    ax.set_title("Speedup Comparison: Parallel vs SIMD")
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Speedup")
    ax.legend()
    ax.grid(True)

    # Aggiungi una linea di riferimento ideale y=x
    max_threads = max(thread_list)
    x_ideal = np.arange(1, max_threads + 1)
    y_ideal = x_ideal
    ax.plot(x_ideal, y_ideal, 'k--', alpha=0.5, label='Ideal Speedup')

    # Salva il grafico
    plt.tight_layout()
    plt.savefig("../result/speedup_parallel_vs_simd.png")
    print("Saved plot as speedup_parallel_vs_simd.png")

    # 2. Confronto tra speedup_hll e speedup_hll_simd
    fig, ax = plt.subplots(figsize=(10, 6))

    for cat in categories:
        color = categories_with_colors[cat]
        hll_values = np.array(avg_values_dict["speedup_hll"][cat])
        hll_simd_values = np.array(avg_values_dict["speedup_hll_simd"][cat])

        # Crea liste filtrate per il grafico
        x_hll = []
        y_hll = []
        x_hll_simd = []
        y_hll_simd = []

        for i in range(len(thread_list)):
            if i < len(hll_values) and not np.isnan(hll_values[i]):
                x_hll.append(thread_list[i])
                y_hll.append(hll_values[i])

            if i < len(hll_simd_values) and not np.isnan(hll_simd_values[i]):
                x_hll_simd.append(thread_list[i])
                y_hll_simd.append(hll_simd_values[i])

        if x_hll:  # Grafica solo se abbiamo dati
            ax.plot(x_hll, y_hll, marker='o', linestyle='-', label=f"{cat} (HLL)", color=color)

        if x_hll_simd:  # Grafica solo se abbiamo dati
            ax.plot(x_hll_simd, y_hll_simd, marker='s', linestyle='--', label=f"{cat} (HLL SIMD)", color=color, alpha=0.7)

    # Configura il grafico
    ax.set_title("Speedup Comparison: HLL vs HLL SIMD")
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Speedup")
    ax.legend()
    ax.grid(True)

    # Aggiungi una linea di riferimento ideale y=x
    ax.plot(x_ideal, y_ideal, 'k--', alpha=0.5, label='Ideal Speedup')

    # Salva il grafico
    plt.tight_layout()
    plt.savefig("../result/speedup_hll_vs_hll_simd.png")
    print("Saved plot as speedup_hll_vs_hll_simd.png")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]

    data = parse_csv_data(csv_file)
    matrix_categories, categories_with_colors, matrix_sizes = categorize_matrices(data)

    # Conta le matrici in ogni categoria
    for cat, matrices in matrix_categories.items():
        print(f"Category {cat}: {len(matrices)} matrices")

    # Metriche di speedup da analizzare
    speedup_metrics = [
        "speedup_parallel",
        "speedup_simd",
        "speedup_hll",
        "speedup_hll_simd"
    ]

    # Salva tutti i valori medi
    all_avg_values = {}
    thread_list = None

    # Calcola le medie per ogni metrica
    for metric in speedup_metrics:
        print(f"\n{'='*50}")
        print(f"ANALYSIS FOR {metric}")
        print(f"{'='*50}")
        avg_values, threads = print_speedup_by_category(data, matrix_categories, metric)
        all_avg_values[metric] = avg_values
        if thread_list is None:
            thread_list = threads

    # Crea grafici individuali per ogni metrica
    plot_individual_speedup(all_avg_values, thread_list, categories_with_colors)

    # Crea grafici di confronto
    plot_comparison_charts(all_avg_values, thread_list, categories_with_colors)

    # Analisi aggiuntiva: Calcola il fattore di miglioramento tra le diverse implementazioni
    print("\nRapporto di miglioramento tra le diverse implementazioni:")
    print("-" * 50)
    for cat in matrix_categories:
        print(f"\nCategoria: {cat}")
        # SIMD vs. Parallel di base
        simd_vs_parallel = []
        hll_vs_parallel = []
        hll_simd_vs_hll = []

        for i, t in enumerate(thread_list):
            if (i < len(all_avg_values["speedup_parallel"][cat]) and
                    i < len(all_avg_values["speedup_simd"][cat]) and
                    i < len(all_avg_values["speedup_hll"][cat]) and
                    i < len(all_avg_values["speedup_hll_simd"][cat])):

                parallel = all_avg_values["speedup_parallel"][cat][i]
                simd = all_avg_values["speedup_simd"][cat][i]
                hll = all_avg_values["speedup_hll"][cat][i]
                hll_simd = all_avg_values["speedup_hll_simd"][cat][i]

                if not np.isnan(parallel) and not np.isnan(simd) and parallel != 0:
                    simd_vs_parallel.append(simd / parallel)

                if not np.isnan(parallel) and not np.isnan(hll) and parallel != 0:
                    hll_vs_parallel.append(hll / parallel)

                if not np.isnan(hll) and not np.isnan(hll_simd) and hll != 0:
                    hll_simd_vs_hll.append(hll_simd / hll)

        if simd_vs_parallel:
            print(f"  SIMD vs. Parallel: {np.mean(simd_vs_parallel):.2f}x miglioramento")
        if hll_vs_parallel:
            print(f"  HLL vs. Parallel: {np.mean(hll_vs_parallel):.2f}x miglioramento")
        if hll_simd_vs_hll:
            print(f"  HLL SIMD vs. HLL: {np.mean(hll_simd_vs_hll):.2f}x miglioramento")

    print("\nAnalisi completata!")

if __name__ == "__main__":
    main()