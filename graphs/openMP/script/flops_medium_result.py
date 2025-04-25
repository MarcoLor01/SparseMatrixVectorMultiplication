import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = "../result"
def parse_csv_data(file_path):
    """Parse il file CSV con i dati di performance delle matrici."""
    data = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def categorize_matrices(data):
    """Categorizza le matrici per dimensione basata sui valori nonzeros."""
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

def analyze_flops_by_category(data, matrix_categories):
    """Analizza i valori FLOPS per categoria di matrice e numero di thread."""
    # Colonne FLOPS da analizzare
    flops_columns = [
        "flops_parallel",
        "flops_parallel_simd",
        "flops_parallel_hll",
        "flops_parallel_hll_simd"
    ]

    # Nomi più leggibili per i grafici
    column_display_names = {
        "flops_parallel": "CSR",
        "flops_parallel_simd": "CSR SIMD",
        "flops_parallel_hll": "HLL",
        "flops_parallel_hll_simd": "HLL SIMD"
    }

    # Raggruppa dati per categoria e numero di thread
    results = {col: {cat: {} for cat in matrix_categories} for col in flops_columns}
    thread_counts = set()

    for row in data:
        matrix_name = row['matrix_name']
        num_threads = int(row['num_threads'])
        thread_counts.add(num_threads)

        # Trova a quale categoria appartiene questa matrice
        for cat, matrices in matrix_categories.items():
            if matrix_name in matrices:
                # Per ciascuna colonna FLOPS, raccogli i dati
                for col in flops_columns:
                    if row[col] and row[col] != '':
                        flops = float(row[col])
                        if num_threads not in results[col][cat]:
                            results[col][cat][num_threads] = []
                        results[col][cat][num_threads].append(flops)
                break

    # Ordina i numeri di thread
    thread_list = sorted(thread_counts)

    # Calcola medie per ogni combinazione di categoria, colonna FLOPS e numero di thread
    avg_results = {col: {cat: [] for cat in matrix_categories} for col in flops_columns}

    # Stampa statistiche per ogni colonna FLOPS
    for col in flops_columns:
        print(f"\n=== Statistiche per {column_display_names[col]} ===")
        header = "Categoria".ljust(15)
        for t in thread_list:
            header += f"{t} threads".ljust(15)
        print(header)
        print("-" * len(header))

        for cat in matrix_categories:
            row = cat.ljust(15)
            for t in thread_list:
                if t in results[col][cat] and results[col][cat][t]:
                    avg = np.mean(results[col][cat][t])
                    avg_results[col][cat].append(avg)
                    row += f"{avg/1e6:.2f}M".ljust(15)
                else:
                    avg_results[col][cat].append(float('nan'))
                    row += "-".ljust(15)
            print(row)

    return avg_results, thread_list, column_display_names

def plot_individual_flops(avg_results, thread_list, categories_with_colors, column_display_names):
    """Crea grafici individuali per ogni metrica FLOPS con tutte le categorie."""
    columns = list(avg_results.keys())
    categories = list(categories_with_colors.keys())

    # Crea un grafico separato per ogni metrica FLOPS
    for col in columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        for cat in categories:
            color = categories_with_colors[cat]
            values = avg_results[col][cat]

            # Filtra valori NaN
            x_values = []
            y_values = []
            for i, val in enumerate(values):
                if not np.isnan(val):
                    x_values.append(thread_list[i])
                    y_values.append(val / 1e6)  # Converti in milioni

            if x_values:  # Solo se abbiamo dati
                ax.plot(x_values, y_values, marker='o', label=cat, color=color)

        # Configura il grafico
        display_name = column_display_names[col]
        ax.set_title(f"FLOPS {display_name} per Numero di Thread")
        ax.set_xlabel("Numero di Thread")
        ax.set_ylabel("FLOPS (milioni)")
        ax.legend()
        ax.grid(True)

        # Salva il grafico
        plt.tight_layout()

        filename = f"{col}_by_category.png"
        filename = os.path.join(output_dir, filename)
        plt.savefig(filename)
        print(f"Salvato grafico come {filename}")
        plt.close(fig)

def plot_bar_comparison(avg_results, thread_list, categories_with_colors, column_display_names):
    """Crea grafici a barre per confrontare tutte e 4 le implementazioni per numero di thread."""
    categories = list(categories_with_colors.keys())
    columns = list(avg_results.keys())

    # Colori per le diverse implementazioni
    column_colors = {
        "flops_parallel": "cornflowerblue",
        "flops_parallel_simd": "royalblue",
        "flops_parallel_hll": "lightcoral",
        "flops_parallel_hll_simd": "firebrick"
    }

    # Un grafico per ogni numero di thread
    for t_idx, t in enumerate(thread_list):
        # Verifica se abbiamo dati per questo thread
        has_thread_data = False
        for cat in categories:
            for col in columns:
                if t_idx < len(avg_results[col][cat]) and not np.isnan(avg_results[col][cat][t_idx]):
                    has_thread_data = True
                    break
            if has_thread_data:
                break

        if not has_thread_data:
            print(f"Nessun dato valido per {t} thread, salto questo grafico")
            continue

        # Crea il grafico
        fig, ax = plt.subplots(figsize=(14, 8))

        # Filtra le categorie che hanno dati per questo thread
        valid_categories = []
        for cat in categories:
            has_cat_data = False
            for col in columns:
                if t_idx < len(avg_results[col][cat]) and not np.isnan(avg_results[col][cat][t_idx]):
                    has_cat_data = True
                    break
            if has_cat_data:
                valid_categories.append(cat)

        # Configurazione barre
        n_categories = len(valid_categories)
        n_columns = len(columns)
        bar_width = 0.2
        group_width = n_columns * bar_width

        # Posizioni delle barre
        cat_positions = np.arange(n_categories)

        # Plotta barre per ogni metrica
        for c_idx, col in enumerate(columns):
            values = []
            for cat in valid_categories:
                if t_idx < len(avg_results[col][cat]) and not np.isnan(avg_results[col][cat][t_idx]):
                    values.append(avg_results[col][cat][t_idx] / 1e6)  # Converti in milioni
                else:
                    values.append(0)

            # Calcola posizione barre
            bar_pos = cat_positions - group_width/2 + (c_idx + 0.5) * bar_width

            # Plotta barre
            ax.bar(bar_pos, values, width=bar_width,
                   label=column_display_names[col],
                   color=column_colors[col],
                   alpha=0.8)

        # Etichette e titolo
        ax.set_title(f"Confronto Prestazioni con {t} Thread")
        ax.set_xlabel("Categoria Matrice")
        ax.set_ylabel("FLOPS (milioni)")
        ax.set_xticks(cat_positions)
        ax.set_xticklabels(valid_categories, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Salva grafico
        plt.tight_layout()
        filename = f"comparison_threads_{t}.png"
        filename = os.path.join(output_dir, filename)
        plt.savefig(filename)
        print(f"Salvato grafico di confronto per {t} thread come {filename}")
        plt.close(fig)

    # Facciamo anche un grafico per categoria, per vedere tutte le implementazioni insieme
    for cat in categories:
        # Verifica se abbiamo dati per questa categoria
        has_cat_data = False
        for col in columns:
            for t_idx in range(len(thread_list)):
                if t_idx < len(avg_results[col][cat]) and not np.isnan(avg_results[col][cat][t_idx]):
                    has_cat_data = True
                    break
            if has_cat_data:
                break

        if not has_cat_data:
            print(f"Nessun dato valido per categoria {cat}, salto questo grafico")
            continue

        # Trova i thread con dati validi
        valid_threads = []
        for t_idx, t in enumerate(thread_list):
            has_valid_data = False
            for col in columns:
                if t_idx < len(avg_results[col][cat]) and not np.isnan(avg_results[col][cat][t_idx]):
                    has_valid_data = True
                    break
            if has_valid_data:
                valid_threads.append((t_idx, t))

        # Crea il grafico
        fig, ax = plt.subplots(figsize=(12, 8))

        # Configurazione barre
        n_threads = len(valid_threads)
        n_columns = len(columns)
        bar_width = 0.2
        group_width = n_columns * bar_width

        # Posizioni delle barre
        thread_positions = np.arange(n_threads)

        # Plotta barre per ogni metrica
        for c_idx, col in enumerate(columns):
            values = []
            for t_idx, t in valid_threads:
                if t_idx < len(avg_results[col][cat]) and not np.isnan(avg_results[col][cat][t_idx]):
                    values.append(avg_results[col][cat][t_idx] / 1e6)  # Converti in milioni
                else:
                    values.append(0)

            # Calcola posizione barre
            bar_pos = thread_positions - group_width/2 + (c_idx + 0.5) * bar_width

            # Plotta barre
            ax.bar(bar_pos, values, width=bar_width,
                   label=column_display_names[col],
                   color=column_colors[col],
                   alpha=0.8)

        # Etichette e titolo
        ax.set_title(f"Confronto Prestazioni per Categoria: {cat}")
        ax.set_xlabel("Numero di Thread")
        ax.set_ylabel("FLOPS (milioni)")
        ax.set_xticks(thread_positions)
        ax.set_xticklabels([t for _, t in valid_threads])
        ax.legend()
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Salva grafico
        plt.tight_layout()
        filename = f"comparison_category_{cat.replace(' ', '_')}.png"
        filename = os.path.join(output_dir, filename)
        plt.savefig(filename)
        print(f"Salvato grafico di confronto per categoria {cat} come {filename}")
        plt.close(fig)

def plot_csr_vs_simd(avg_results, thread_list, categories_with_colors):
    """Crea un grafico con FLOPS CSR (linea piena) e CSR SIMD (linea tratteggiata) per categoria."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for cat, color in categories_with_colors.items():
        csr_values = avg_results["flops_parallel"][cat]
        simd_values = avg_results["flops_parallel_simd"][cat]

        x = []
        y_csr = []
        y_simd = []

        for i, t in enumerate(thread_list):
            v_csr = csr_values[i] if i < len(csr_values) else np.nan
            v_simd = simd_values[i] if i < len(simd_values) else np.nan

            if not np.isnan(v_csr) and not np.isnan(v_simd):
                x.append(t)
                y_csr.append(v_csr / 1e6)
                y_simd.append(v_simd / 1e6)

        if x:
            ax.plot(x, y_csr, label=f"{cat} (CSR)", color=color, linestyle='-', marker='o')
            ax.plot(x, y_simd, label=f"{cat} (CSR SIMD)", color=color, linestyle='--', marker='o')

    ax.set_title("Confronto FLOPS: CSR vs CSR SIMD per Categoria")
    ax.set_xlabel("Numero di Thread")
    ax.set_ylabel("FLOPS (milioni)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    filename = os.path.join(output_dir, "flops_csr_vs_simd.png")
    plt.savefig(filename)
    print(f"Salvato grafico comparativo CSR vs CSR SIMD come {filename}")
    plt.close(fig)


def plot_hll_vs_simd(avg_results, thread_list, categories_with_colors):
    """Crea un grafico con FLOPS HLL (linea piena) e HLL SIMD (linea tratteggiata) per categoria."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for cat, color in categories_with_colors.items():
        csr_values = avg_results["flops_parallel_hll"][cat]
        simd_values = avg_results["flops_parallel_hll_simd"][cat]

        x = []
        y_csr = []
        y_simd = []

        for i, t in enumerate(thread_list):
            v_csr = csr_values[i] if i < len(csr_values) else np.nan
            v_simd = simd_values[i] if i < len(simd_values) else np.nan

            if not np.isnan(v_csr) and not np.isnan(v_simd):
                x.append(t)
                y_csr.append(v_csr / 1e6)
                y_simd.append(v_simd / 1e6)

        if x:
            ax.plot(x, y_csr, label=f"{cat} (HLL)", color=color, linestyle='-', marker='o')
            ax.plot(x, y_simd, label=f"{cat} (HLL SIMD)", color=color, linestyle='--', marker='o')

    ax.set_title("Confronto FLOPS: HLL vs HLL SIMD per Categoria")
    ax.set_xlabel("Numero di Thread")
    ax.set_ylabel("FLOPS (milioni)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    filename = os.path.join(output_dir, "flops_hll_vs_simd.png")
    plt.savefig(filename)
    print(f"Salvato grafico comparativo HLL vs HLL SIMD come {filename}")
    plt.close(fig)

def main():
    if len(sys.argv) < 2:
        print("Utilizzo: python script.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]

    # Carica e categorizza i dati
    data = parse_csv_data(csv_file)
    matrix_categories, categories_with_colors, matrix_sizes = categorize_matrices(data)

    # Stampa il numero di matrici per categoria
    print("=== Distribuzione delle matrici per categoria ===")
    for cat, matrices in matrix_categories.items():
        print(f"Categoria {cat}: {len(matrices)} matrici")

    # Analizza i valori FLOPS
    avg_results, thread_list, column_display_names = analyze_flops_by_category(data, matrix_categories)

    # Crea i grafici individuali per ogni implementazione
    plot_individual_flops(avg_results, thread_list, categories_with_colors, column_display_names)

    # Crea i grafici a barre per confrontare le implementazioni
    plot_bar_comparison(avg_results, thread_list, categories_with_colors, column_display_names)

    # Crea grafico comparativo tra CSR e CSR SIMD
    plot_csr_vs_simd(avg_results, thread_list, categories_with_colors)

    # Crea grafico comparativo tra HLL e HLL SIMD
    plot_hll_vs_simd(avg_results, thread_list, categories_with_colors)

    print("\nAnalisi completata! Tutti i grafici sono stati generati.")

if __name__ == "__main__":
    main()