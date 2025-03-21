import pandas as pd
import matplotlib.pyplot as plt

def plot_flops_vs_threads(csv_file):
    # Legge i dati dal file CSV
    df = pd.read_csv(csv_file)

    # Converti FLOPS in MegaFLOPS
    df['flops_parallel'] /= 1e6
    df['flops_parallel_simd'] /= 1e6

    # Ottiene l'elenco delle matrici
    matrices = df[['matrix_name', 'nonzeros']].drop_duplicates()

    # Crea due grafici separati
    plt.figure(figsize=(12, 5))

    # Grafico per flops_parallel
    plt.subplot(1, 2, 1)
    for _, row in matrices.iterrows():
        matrix = row['matrix_name']
        nz = row['nonzeros']
        subset = df[df['matrix_name'] == matrix]
        plt.plot(subset['num_threads'], subset['flops_parallel'], marker='o', label=f"{matrix} (nz={nz})")
    plt.xlabel('Numero di Thread')
    plt.ylabel('MegaFLOPS (Parallel)')
    plt.title('Flops Parallel vs Thread')
    plt.legend()

    # Grafico per flops_parallel_simd
    plt.subplot(1, 2, 2)
    for _, row in matrices.iterrows():
        matrix = row['matrix_name']
        nz = row['nonzeros']
        subset = df[df['matrix_name'] == matrix]
        plt.plot(subset['num_threads'], subset['flops_parallel_simd'], marker='o', label=f"{matrix} (nz={nz})")
    plt.xlabel('Numero di Thread')
    plt.ylabel('MegaFLOPS (Parallel SIMD)')
    plt.title('Flops Parallel SIMD vs Thread')
    plt.legend()

    plt.tight_layout()
    plt.show()



# Esempio di utilizzo
plot_flops_vs_threads('../../../result/spmv_results_only_big.csv')
