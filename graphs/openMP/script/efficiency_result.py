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

def print_efficiency_by_category(data, matrix_categories, efficiency_column):
    """Print efficiency values from the specified column, grouped by category."""
    # Group data by category and thread count
    category_thread_efficiency = {cat: {} for cat in matrix_categories}

    for row in data:
        matrix_name = row['matrix_name']
        num_threads = int(row['num_threads'])
        if row[efficiency_column] and row[efficiency_column] != '':
            efficiency = float(row[efficiency_column])
        else:
            continue

        # Find which category this matrix belongs to
        for cat, matrices in matrix_categories.items():
            if matrix_name in matrices:
                if num_threads not in category_thread_efficiency[cat]:
                    category_thread_efficiency[cat][num_threads] = []
                category_thread_efficiency[cat][num_threads].append(efficiency)
                break

    # Calculate average efficiency for each category and thread count
    all_threads = set()
    for cat_data in category_thread_efficiency.values():
        all_threads.update(cat_data.keys())
    thread_list = sorted(all_threads)

    # Print header
    header = "Category".ljust(15)
    for t in thread_list:
        header += f"{t} threads".ljust(15)
    print(f"\nAverage {efficiency_column} by category and thread count:")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Print data rows and collect average values
    avg_values = {}
    for cat, thread_data in category_thread_efficiency.items():
        row = cat.ljust(15)
        avg_values[cat] = []

        for t in thread_list:
            if t in thread_data and thread_data[t]:
                avg = np.mean(thread_data[t])
                avg_values[cat].append(avg)
                row += f"{avg:.4f}".ljust(15)
            else:
                avg_values[cat].append(float('nan'))
                row += "-".ljust(15)
        print(row)

    print("-" * len(header))

    # Also print the raw values for easier copy-paste
    print(f"\nRaw values for {efficiency_column} (for plotting):")
    for cat in matrix_categories:
        print(f"{cat}:", end=" ")
        print(", ".join([f"{v}" for v in avg_values[cat]]))

    return avg_values, thread_list

def plot_individual_efficiency(avg_values_dict, thread_list, categories_with_colors):
    """Create individual plots for each efficiency metric with all categories."""
    metrics = list(avg_values_dict.keys())
    categories = list(categories_with_colors.keys())

    # Create a separate plot for each metric
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        for cat in categories:
            color = categories_with_colors[cat]
            values = avg_values_dict[metric][cat]

            # Filter out NaN values
            x_values = []
            y_values = []
            for i, val in enumerate(values):
                if not np.isnan(val):
                    x_values.append(thread_list[i])
                    y_values.append(val)

            if x_values:  # Only plot if we have data
                ax.plot(x_values, y_values, marker='o', label=cat, color=color)

        # Add a line for ideal efficiency (y=1)
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Ideal Efficiency')

        # Configure the chart
        ax.set_title(f"{metric} per Thread Count")
        ax.set_xlabel("Number of Threads")
        ax.set_ylabel("Efficiency")
        ax.legend()
        ax.grid(True)

        # Save the chart
        plt.tight_layout()
        plt.savefig(f"../result/{metric}_by_category_efficiency.png")
        print(f"Saved plot as {metric}_by_category.png")

def plot_comparison_charts(avg_values_dict, thread_list, categories_with_colors):
    """Create comparison charts between parallel vs SIMD and HLL vs HLL SIMD."""
    categories = list(categories_with_colors.keys())

    # 1. Parallel vs SIMD
    fig, ax = plt.subplots(figsize=(10, 6))

    for cat in categories:
        color = categories_with_colors[cat]
        parallel_values = np.array(avg_values_dict["efficiency_parallel"][cat])
        simd_values = np.array(avg_values_dict["efficiency_simd"][cat])

        # Create filtered lists for plotting
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

        if x_parallel:  # Only plot if we have data
            ax.plot(x_parallel, y_parallel, marker='o', linestyle='-', label=f"{cat} (Parallel)", color=color)

        if x_simd:  # Only plot if we have data
            ax.plot(x_simd, y_simd, marker='s', linestyle='--', label=f"{cat} (SIMD)", color=color, alpha=0.7)

    # Add a line for ideal efficiency (y=1)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Ideal Efficiency')

    # Configure the chart
    ax.set_title("Efficiency Comparison: Parallel vs SIMD")
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Efficiency")
    ax.legend()
    ax.grid(True)

    # Save the chart
    plt.tight_layout()
    plt.savefig("../result/comparison_parallel_vs_simd_efficiency.png")
    print("Saved plot as comparison_parallel_vs_simd.png")

    # 2. HLL vs HLL SIMD
    fig, ax = plt.subplots(figsize=(10, 6))

    for cat in categories:
        color = categories_with_colors[cat]
        hll_values = np.array(avg_values_dict["efficiency_hll"][cat])
        hll_simd_values = np.array(avg_values_dict["efficiency_hll_simd"][cat])

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

    # Add a line for ideal efficiency (y=1)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Ideal Efficiency')

    # Configure the chart
    ax.set_title("Efficiency Comparison: HLL vs HLL SIMD")
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Efficiency")
    ax.legend()
    ax.grid(True)

    # Save the chart
    plt.tight_layout()
    plt.savefig("../result/comparison_hll_vs_hll_simd_efficiency.png")
    print("Saved plot as comparison_hll_vs_hll_simd.png")

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
    efficiency_metrics = [
        "efficiency_parallel",
        "efficiency_simd",
        "efficiency_hll",
        "efficiency_hll_simd"
    ]

    # Save all the average values
    all_avg_values = {}
    thread_list = None

    # Calculate averages for each metric
    for metric in efficiency_metrics:
        print(f"\n{'='*50}")
        print(f"ANALYSIS FOR {metric}")
        print(f"{'='*50}")
        avg_values, threads = print_efficiency_by_category(data, matrix_categories, metric)
        all_avg_values[metric] = avg_values
        if thread_list is None:
            thread_list = threads

    # Create individual plots for each metric
    plot_individual_efficiency(all_avg_values, thread_list, categories_with_colors)

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
            if (i < len(all_avg_values["efficiency_parallel"][cat]) and
                    i < len(all_avg_values["efficiency_simd"][cat]) and
                    i < len(all_avg_values["efficiency_hll"][cat]) and
                    i < len(all_avg_values["efficiency_hll_simd"][cat])):

                base = all_avg_values["efficiency_parallel"][cat][i]
                simd = all_avg_values["efficiency_simd"][cat][i]
                hll = all_avg_values["efficiency_hll"][cat][i]
                combined = all_avg_values["efficiency_hll_simd"][cat][i]

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