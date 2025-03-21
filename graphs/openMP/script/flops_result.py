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

def print_flops_by_category(data, matrix_categories, flops_column):
    """Print flops values from the specified column, grouped by category."""
    # Group data by category and thread count
    category_thread_flops = {cat: {} for cat in matrix_categories}

    for row in data:
        matrix_name = row['matrix_name']
        num_threads = int(row['num_threads'])
        if row[flops_column] and row[flops_column] != '':
            flops = float(row[flops_column])
        else:
            continue

        # Find which category this matrix belongs to
        for cat, matrices in matrix_categories.items():
            if matrix_name in matrices:
                if num_threads not in category_thread_flops[cat]:
                    category_thread_flops[cat][num_threads] = []
                category_thread_flops[cat][num_threads].append(flops)
                break

    # Calculate average flops for each category and thread count
    all_threads = set()
    for cat_data in category_thread_flops.values():
        all_threads.update(cat_data.keys())
    thread_list = sorted(all_threads)

    # Print header
    header = "Category".ljust(15)
    for t in thread_list:
        header += f"{t} threads".ljust(15)
    print(f"\nAverage {flops_column} (in millions) by category and thread count:")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Print data rows and collect average values
    avg_values = {}
    for cat, thread_data in category_thread_flops.items():
        row = cat.ljust(15)
        avg_values[cat] = []

        for t in thread_list:
            if t in thread_data and thread_data[t]:
                avg = np.mean(thread_data[t])
                avg_values[cat].append(avg)
                row += f"{avg/1e6:.2f}M".ljust(15)
            else:
                avg_values[cat].append(float('nan'))
                row += "-".ljust(15)
        print(row)

    print("-" * len(header))

    # Also print the raw values for easier copy-paste
    print(f"\nRaw values for {flops_column} (for plotting):")
    for cat in matrix_categories:
        print(f"{cat}:", end=" ")
        print(", ".join([f"{v}" for v in avg_values[cat]]))

    return avg_values, thread_list

def plot_individual_flops(avg_values_dict, thread_list, categories_with_colors):
    """Create individual plots for each FLOPS metric with all categories."""
    metrics = list(avg_values_dict.keys())
    categories = list(categories_with_colors.keys())

    # Create a separate plot for each metric
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        for cat in categories:
            color = categories_with_colors[cat]
            values = avg_values_dict[metric][cat]

            # Convert values to millions and filter out NaN values
            x_values = []
            y_values = []
            for i, val in enumerate(values):
                if not np.isnan(val):
                    x_values.append(thread_list[i])
                    y_values.append(val / 1e6)

            if x_values:  # Only plot if we have data
                ax.plot(x_values, y_values, marker='o', label=cat, color=color)

        # Configure the chart
        ax.set_title(f"{metric} per Thread Count")
        ax.set_xlabel("Number of Threads")
        ax.set_ylabel("FLOPS (millions)")
        ax.legend()
        ax.grid(True)

        # Save the chart
        plt.tight_layout()
        plt.savefig(f"../result/{metric}_by_category.png")
        print(f"Saved plot as {metric}_by_category.png")

def plot_comparison_charts(avg_values_dict, thread_list, categories_with_colors):
    """Create comparison charts between normal vs SIMD and HLL normal vs HLL SIMD."""
    categories = list(categories_with_colors.keys())

    # 1. CSR normale vs CSR SIMD
    fig, ax = plt.subplots(figsize=(10, 6))

    for cat in categories:
        color = categories_with_colors[cat]
        normal_values = np.array(avg_values_dict["flops_parallel"][cat]) / 1e6
        simd_values = np.array(avg_values_dict["flops_parallel_simd"][cat]) / 1e6

        # Create filtered lists for plotting
        x_normal = []
        y_normal = []
        x_simd = []
        y_simd = []

        for i in range(len(thread_list)):
            if i < len(normal_values) and not np.isnan(normal_values[i]):
                x_normal.append(thread_list[i])
                y_normal.append(normal_values[i])

            if i < len(simd_values) and not np.isnan(simd_values[i]):
                x_simd.append(thread_list[i])
                y_simd.append(simd_values[i])

        if x_normal:  # Only plot if we have data
            ax.plot(x_normal, y_normal, marker='o', linestyle='-', label=f"{cat} (Normal)", color=color)

        if x_simd:  # Only plot if we have data
            ax.plot(x_simd, y_simd, marker='s', linestyle='--', label=f"{cat} (SIMD)", color=color, alpha=0.7)

    # Configure the chart
    ax.set_title("FLOPS CSR vs CSR with SIMD istructions")
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("FLOPS (millions)")
    ax.legend()
    ax.grid(True)

    # Save the chart
    plt.tight_layout()
    plt.savefig("../result/csr_normal_vs_simd.png")
    print("Saved plot as csr_normal_vs_simd.png")

    # 2. HLL normale vs HLL SIMD
    fig, ax = plt.subplots(figsize=(10, 6))

    for cat in categories:
        color = categories_with_colors[cat]
        hll_values = np.array(avg_values_dict["flops_parallel_hll"][cat]) / 1e6
        hll_simd_values = np.array(avg_values_dict["flops_parallel_hll_simd"][cat]) / 1e6

        # Create filtered lists for plotting
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

        if x_hll:  # Only plot if we have data
            ax.plot(x_hll, y_hll, marker='o', linestyle='-', label=f"{cat} (HLL)", color=color)

        if x_hll_simd:  # Only plot if we have data
            ax.plot(x_hll_simd, y_hll_simd, marker='s', linestyle='--', label=f"{cat} (HLL SIMD)", color=color, alpha=0.7)

    # Configure the chart
    ax.set_title("HLL Normal vs HLL SIMD")
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("FLOPS (millions)")
    ax.legend()
    ax.grid(True)

    # Save the chart
    plt.tight_layout()
    plt.savefig("../result/hll_normal_vs_simd.png")
    print("Saved plot as hll_normal_vs_simd.png")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]

    data = parse_csv_data(csv_file)
    matrix_categories, categories_with_colors, matrix_sizes = categorize_matrices(data)

    # Count matrices in each category
    for cat, matrices in matrix_categories.items():
        print(f"Category {cat}: {len(matrices)} matrices")

    # Metrics to analyze
    flops_metrics = [
        "flops_parallel",
        "flops_parallel_simd",
        "flops_parallel_hll",
        "flops_parallel_hll_simd"
    ]

    # Save all the average values
    all_avg_values = {}
    thread_list = None

    # Calculate averages for each metric
    for metric in flops_metrics:
        print(f"\n{'='*50}")
        print(f"ANALYSIS FOR {metric}")
        print(f"{'='*50}")
        avg_values, threads = print_flops_by_category(data, matrix_categories, metric)
        all_avg_values[metric] = avg_values
        if thread_list is None:
            thread_list = threads

    # Create individual plots for each metric
    plot_individual_flops(all_avg_values, thread_list, categories_with_colors)

    # Create comparison charts
    plot_comparison_charts(all_avg_values, thread_list, categories_with_colors)

    # Additional analysis: Calculate the improvement factor for each optimization
    print("\nImprovement factors for different optimizations:")
    print("-" * 50)
    for cat in matrix_categories:
        print(f"\nCategory: {cat}")
        # SIMD vs. basic parallel
        simd_vs_parallel = []
        hll_vs_parallel = []
        combined_vs_parallel = []

        for i, t in enumerate(thread_list):
            if (i < len(all_avg_values["flops_parallel"][cat]) and
                    i < len(all_avg_values["flops_parallel_simd"][cat]) and
                    i < len(all_avg_values["flops_parallel_hll"][cat]) and
                    i < len(all_avg_values["flops_parallel_hll_simd"][cat])):

                base = all_avg_values["flops_parallel"][cat][i]
                simd = all_avg_values["flops_parallel_simd"][cat][i]
                hll = all_avg_values["flops_parallel_hll"][cat][i]
                combined = all_avg_values["flops_parallel_hll_simd"][cat][i]

                if not np.isnan(base) and not np.isnan(simd) and base != 0:
                    simd_vs_parallel.append(simd / base)

                if not np.isnan(base) and not np.isnan(hll) and base != 0:
                    hll_vs_parallel.append(hll / base)

                if not np.isnan(base) and not np.isnan(combined) and base != 0:
                    combined_vs_parallel.append(combined / base)

        if simd_vs_parallel:
            print(f"  SIMD vs. Parallel: {np.mean(simd_vs_parallel):.2f}x improvement")
        if hll_vs_parallel:
            print(f"  HLL vs. Parallel: {np.mean(hll_vs_parallel):.2f}x improvement")
        if combined_vs_parallel:
            print(f"  Combined vs. Parallel: {np.mean(combined_vs_parallel):.2f}x improvement")

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()