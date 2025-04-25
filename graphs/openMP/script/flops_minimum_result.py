import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

def parse_csv_data(file_path):
    """Parse the CSV file with matrix performance data."""
    data = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def categorize_matrices(data):
    """Categorize matrices by size based on nonzeros with updated categories."""
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
        # Assicurati che la colonna 'nonzeros' esista e non sia vuota
        if 'nonzeros' in row and row['nonzeros']:
            try:
                nonzeros = int(row['nonzeros'])
                matrix_sizes[matrix_name] = nonzeros
            except ValueError:
                print(f"Warning: Could not parse nonzeros for matrix {matrix_name}. Skipping.")
                continue # Salta questa riga se non puoi leggere nonzeros
        else:
            print(f"Warning: Missing 'nonzeros' for matrix {matrix_name}. Skipping categorization.")
            continue # Salta se manca la colonna nonzeros


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

def calculate_flops_in_millions(nonzeros, time_seconds):
    """Calculate FLOPS (floating-point operations per second) and return the value in Millions."""
    # Formula: 2 * NZ operations per SpMV.
    # Result is FLOPS = 2 * NZ / time_seconds.
    # Divide by 1e6 to get Millions of FLOPS.
    if nonzeros is None or time_seconds is None or time_seconds <= 0:
        return float('nan')
    try:
        # Assicurati che nonzeros sia numerico
        nonzeros_float = float(nonzeros)
        return (2.0 * nonzeros_float / time_seconds) / 1e6
    except ValueError:
        print(f"Warning: Could not convert nonzeros '{nonzeros}' to float during calculation.")
        return float('nan')


def process_min_times_to_flops_millions(data, matrix_categories, matrix_sizes, time_metrics):
    """Process minimum time metrics and convert to FLOPS (in millions)."""
    # Group data by category and thread count
    category_thread_flops_millions = {metric: {cat: {} for cat in matrix_categories} for metric in time_metrics}

    for row in data:
        matrix_name = row['matrix_name']
        # Salta se la matrice non ha un numero di nonzeros valido
        if matrix_name not in matrix_sizes:
            continue
        nonzeros = matrix_sizes[matrix_name]

        # Salta se non c'è num_threads o non è valido
        if 'num_threads' not in row or not row['num_threads']:
            # print(f"Warning: Missing or empty 'num_threads' for matrix {matrix_name}. Skipping row.")
            continue
        try:
            num_threads = int(row['num_threads'])
        except ValueError:
            # print(f"Warning: Invalid 'num_threads' value '{row['num_threads']}' for matrix {matrix_name}. Skipping row.")
            continue


        # Find which category this matrix belongs to
        category = None
        for cat, matrices in matrix_categories.items():
            if matrix_name in matrices:
                category = cat
                break

        if not category:
            continue

        # Process each time metric and convert to FLOPS (millions)
        for metric in time_metrics:
            # Controlla se la metrica esiste nella riga e non è vuota
            if metric in row and row[metric]:
                try:
                    time_ms = float(row[metric])
                    if time_ms > 0:
                        time_seconds = time_ms / 1000.0  # Convert to seconds
                        flops_millions = calculate_flops_in_millions(nonzeros, time_seconds)

                        if num_threads not in category_thread_flops_millions[metric][category]:
                            category_thread_flops_millions[metric][category][num_threads] = []

                        category_thread_flops_millions[metric][category][num_threads].append(flops_millions)
                except ValueError:
                    # Ignora valori non numerici per il tempo
                    # print(f"Warning: Invalid time value '{row[metric]}' for metric {metric}, matrix {matrix_name}. Skipping value.")
                    pass # Non fare nulla se il tempo non è valido

    # Calculate average FLOPS (millions) for each category, metric, and thread count
    all_threads = set()
    for metric_data in category_thread_flops_millions.values():
        for cat_data in metric_data.values():
            all_threads.update(cat_data.keys())
    thread_list = sorted(all_threads)

    # Prepare results
    avg_values = {metric: {cat: [] for cat in matrix_categories} for metric in time_metrics}

    # Print headers and results for each metric - FORMATO SIMILE AL PRIMO SCRIPT
    for metric in time_metrics:
        # Usa i nomi brevi per le intestazioni se disponibili
        display_name = metric.replace("min_value_", "").upper() # es. CSR, HLL, CSR_SIMD
        print(f"\nAverage FLOPS (in millions) for {display_name} (based on min times) by category and thread count:") # Modifica titolo
        print("-" * 80)

        header = "Category".ljust(15)
        for t in thread_list:
            header += f"{t} threads".ljust(15)
        print(header)
        print("-" * 80)

        for cat in matrix_categories:
            row_str = cat.ljust(15) # Rinomina variabile per chiarezza
            avg_values[metric][cat] = [] # Inizializza qui
            for t in thread_list:
                # Verifica se la chiave 't' esiste ed ha valori non vuoti/NaN
                if t in category_thread_flops_millions[metric][cat] and category_thread_flops_millions[metric][cat][t]:
                    # Filtra NaN prima di calcolare la media
                    valid_flops = [f for f in category_thread_flops_millions[metric][cat][t] if not np.isnan(f)]
                    if valid_flops:
                        avg = np.mean(valid_flops)
                        avg_values[metric][cat].append(avg)
                        row_str += f"{avg:.2f}M".ljust(15) # Usa 'M' come suffisso
                    else:
                        avg_values[metric][cat].append(float('nan'))
                        row_str += "-".ljust(15)
                else:
                    avg_values[metric][cat].append(float('nan'))
                    row_str += "-".ljust(15)
            print(row_str)

        print("-" * 80)
        # Stampa valori raw - FORMATO SIMILE AL PRIMO SCRIPT
        print(f"\nRaw values for {display_name} (in Millions of FLOPS, for plotting):")
        for cat in matrix_categories:
            # Assicurati che ci sia una lista per la categoria prima di accedervi
            if cat in avg_values[metric]:
                print(f"{cat}:", end=" ")
                # Gestisci NaN nella stampa raw
                print(", ".join([f"{v:.4f}" if not np.isnan(v) else "nan" for v in avg_values[metric][cat]]))
            else:
                print(f"{cat}: No data")


    return avg_values, thread_list

def plot_individual_flops_millions(avg_values_dict, thread_list, categories_with_colors, metric_display_names):
    """Create individual plots for each FLOPS (millions) metric with all categories."""
    metrics = list(avg_values_dict.keys())
    categories = list(categories_with_colors.keys())

    # Create a separate plot for each metric
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        for cat in categories:
            color = categories_with_colors[cat]
            # Assicurati che la metrica e la categoria esistano nel dizionario
            if metric in avg_values_dict and cat in avg_values_dict[metric]:
                values = avg_values_dict[metric][cat]

                # Filter out NaN values
                x_values = []
                y_values = []
                for i, val in enumerate(values):
                    # Assicurati che thread_list[i] esista
                    if i < len(thread_list) and not np.isnan(val):
                        x_values.append(thread_list[i])
                        y_values.append(val) # Value è già in FLOPS (millions)

                if x_values:  # Only plot if we have data
                    ax.plot(x_values, y_values, marker='o', label=cat, color=color)
            else:
                print(f"Skipping plot for {cat} in metric {metric} due to missing data.")


        # Configure the chart
        display_name = metric_display_names.get(metric, metric.replace("min_value_","").upper())
        ax.set_title(f"{display_name} Performance per Thread Count")
        ax.set_xlabel("Number of Threads")
        ax.set_ylabel("FLOPS (millions)") # <<< MODIFICA ETICHETTA Y
        ax.legend()
        ax.grid(True)

        # Save the chart
        plt.tight_layout()
        # Usa nomi file consistenti
        filename_base = metric.replace("min_value_", "")
        filename = f"{filename_base}_flops_millions_by_category.png"
        try:
            plt.savefig(f"../result/{filename}")
            print(f"Saved plot as {filename}")
        except FileNotFoundError:
            print(f"Error: Directory '../result/' not found. Could not save {filename}")
        plt.close(fig) # Chiudi la figura dopo il salvataggio


def plot_comparison_charts_flops_millions(avg_values_dict, thread_list, categories_with_colors, metric_display_names):
    """Create comparison charts between implementations using FLOPS (millions)."""
    categories = list(categories_with_colors.keys())

    # Coppie di metriche da confrontare e nomi file
    comparisons = [
        ("min_value_csr", "min_value_csr_simd", "CSR vs CSR SIMD", "csr_vs_csr_simd_flops_millions.png"),
        ("min_value_hll", "min_value_hll_simd", "HLL vs HLL SIMD", "hll_vs_hll_simd_flops_millions.png"),
        ("min_value_csr", "min_value_hll", "CSR vs HLL", "csr_vs_hll_flops_millions.png"),
        ("min_value_csr_simd", "min_value_hll_simd", "CSR SIMD vs HLL SIMD", "csr_simd_vs_hll_simd_flops_millions.png")
    ]

    for metric1_key, metric2_key, title_suffix, filename in comparisons:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_successful = False # Flag per tracciare se qualcosa è stato plottato

        metric1_name = metric_display_names.get(metric1_key, metric1_key.replace("min_value_","").upper())
        metric2_name = metric_display_names.get(metric2_key, metric2_key.replace("min_value_","").upper())

        for cat in categories:
            color = categories_with_colors[cat]
            values1 = []
            values2 = []

            # Recupera i dati, gestendo chiavi mancanti
            if metric1_key in avg_values_dict and cat in avg_values_dict[metric1_key]:
                values1 = np.array(avg_values_dict[metric1_key][cat])
            if metric2_key in avg_values_dict and cat in avg_values_dict[metric2_key]:
                values2 = np.array(avg_values_dict[metric2_key][cat])


            # Create filtered lists for plotting
            x1, y1 = [], []
            x2, y2 = [], []

            # Usa la lunghezza minima tra thread_list e i dati disponibili
            max_len = len(thread_list)
            if len(values1) > 0: max_len = min(max_len, len(values1))
            if len(values2) > 0: max_len = min(max_len, len(values2))


            for i in range(max_len):
                if i < len(values1) and not np.isnan(values1[i]):
                    x1.append(thread_list[i])
                    y1.append(values1[i])

                if i < len(values2) and not np.isnan(values2[i]):
                    x2.append(thread_list[i])
                    y2.append(values2[i])


            if x1:  # Only plot if we have data for metric 1
                ax.plot(x1, y1, marker='o', linestyle='-',
                        label=f"{cat} ({metric1_name})", color=color)
                plot_successful = True
            if x2:  # Only plot if we have data for metric 2
                ax.plot(x2, y2, marker='s', linestyle='--',
                        label=f"{cat} ({metric2_name})", color=color, alpha=0.7)
                plot_successful = True

        # Configure the chart only if something was plotted
        if plot_successful:
            ax.set_title(f"{title_suffix} (from min times)")
            ax.set_xlabel("Number of Threads")
            ax.set_ylabel("FLOPS (millions)") # <<< MODIFICA ETICHETTA Y
            ax.legend()
            ax.grid(True)

            # Save the chart
            plt.tight_layout()
            try:
                plt.savefig(f"../result/{filename}")
                print(f"Saved plot as {filename}")
            except FileNotFoundError:
                print(f"Error: Directory '../result/' not found. Could not save {filename}")
        else:
            print(f"Skipping plot {filename} as no data was available for comparison.")

        plt.close(fig) # Chiudi la figura


def calculate_improvement_factors(avg_values_dict, thread_list, matrix_categories):
    """Calculate improvement factors between different implementations based on FLOPS (millions)."""
    print("\nImprovement factors for different implementations (based on min times -> FLOPS millions):") # Modifica titolo
    print("-" * 70)

    # Definisci le coppie per il calcolo del miglioramento
    comparisons = [
        ("CSR SIMD vs. CSR", "min_value_csr_simd", "min_value_csr"),
        ("HLL vs. CSR", "min_value_hll", "min_value_csr"),
        ("HLL SIMD vs. CSR", "min_value_hll_simd", "min_value_csr"),
        ("HLL SIMD vs. HLL", "min_value_hll_simd", "min_value_hll"),
        ("HLL SIMD vs. CSR SIMD", "min_value_hll_simd", "min_value_csr_simd")
    ]

    for cat in matrix_categories:
        print(f"\nCategory: {cat}")
        results_printed = False # Flag per vedere se stampare qualcosa per la categoria

        for label, metric_improved_key, metric_base_key in comparisons:
            ratios = []
            # Verifica che entrambe le metriche e la categoria esistano
            if (metric_improved_key in avg_values_dict and
                    metric_base_key in avg_values_dict and
                    cat in avg_values_dict[metric_improved_key] and
                    cat in avg_values_dict[metric_base_key]):

                values_improved = avg_values_dict[metric_improved_key][cat]
                values_base = avg_values_dict[metric_base_key][cat]
                max_len = min(len(thread_list), len(values_improved), len(values_base))


                for i in range(max_len):
                    improved = values_improved[i]
                    base = values_base[i]

                    if not np.isnan(base) and not np.isnan(improved) and base != 0:
                        ratios.append(improved / base)

            if ratios: # Calcola e stampa solo se ci sono rapporti validi
                mean_improvement = np.mean(ratios)
                print(f"  {label}: {mean_improvement:.2f}x improvement")
                results_printed = True

        if not results_printed:
            print("  No improvement data available for this category.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]

    data = parse_csv_data(csv_file)
    # Gestisci il caso in cui data sia vuoto
    if not data:
        print(f"Error: No data parsed from {csv_file}. Exiting.")
        sys.exit(1)

    matrix_categories, categories_with_colors, matrix_sizes = categorize_matrices(data)
    # Gestisci il caso in cui matrix_sizes sia vuoto (nessuna matrice valida)
    if not matrix_sizes:
        print("Error: No valid matrices found or 'nonzeros' column missing/invalid in CSV. Exiting.")
        sys.exit(1)


    # Count matrices in each category
    print("Matrix counts per category:")
    for cat, matrices in matrix_categories.items():
        print(f"  Category {cat}: {len(matrices)} matrices")

    # Time metrics to analyze and convert to FLOPS (millions)
    time_metrics = [
        "min_value_csr",
        "min_value_hll",
        "min_value_csr_simd",
        "min_value_hll_simd"
    ]

    # Display names for the metrics (for plots and clearer output)
    metric_display_names = {
        "min_value_csr": "CSR",
        "min_value_hll": "HLL",
        "min_value_csr_simd": "CSR SIMD",
        "min_value_hll_simd": "HLL SIMD"
    }

    print(f"\n{'='*50}")
    print(f"CONVERTING MIN TIMES TO FLOPS (Millions) AND ANALYZING") # Modifica titolo
    print(f"{'='*50}")

    # Process min times and convert to FLOPS (millions)
    avg_values, thread_list = process_min_times_to_flops_millions(
        data, matrix_categories, matrix_sizes, time_metrics
    )

    # Verifica se thread_list è stato popolato
    if not thread_list:
        print("\nError: No valid thread data found across metrics. Cannot generate plots or analysis.")
        sys.exit(1)

    # Create individual plots for each metric
    plot_individual_flops_millions(avg_values, thread_list, categories_with_colors, metric_display_names)

    # Create comparison charts
    plot_comparison_charts_flops_millions(avg_values, thread_list, categories_with_colors, metric_display_names)

    # Calculate improvement factors
    calculate_improvement_factors(avg_values, thread_list, matrix_categories)

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()