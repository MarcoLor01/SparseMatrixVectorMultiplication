import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('../../../../result/spmv_results_definitive.csv')
# Definizione dei gruppi basati sul numero di elementi non nulli
def assign_group(nonzeros):
    if nonzeros < 10000:
        return "Gruppo 1: <10K"
    elif nonzeros < 100000:
        return "Gruppo 2: 10K-100K"
    elif nonzeros < 500000:
        return "Gruppo 3: 100K-500K"
    elif nonzeros < 1000000:
        return "Gruppo 4: 500K-1M"
    elif nonzeros < 2500000:
        return "Gruppo 5: 1M-2.5M"
    else:
        return "Gruppo 6: >2.5M"

# Applica la funzione per assegnare ogni matrice al suo gruppo
df['size_group'] = df['nonzeros'].apply(assign_group)

# Riorganizza i dati per il plot
# Crea un DataFrame lungo per seaborn
melted_data = []

# Per CSR standard
for _, row in df.iterrows():
    melted_data.append({
        'matrix_name': row['matrix_name'],
        'size_group': row['size_group'],
        'format': 'CSR',
        'threads': row['num_threads'],
        'flops': row['flops_parallel'],
        'min_flops': row['flops_parallel'] * (row['time_parallel'] / row['min_value_csr']),
        'stddev': row['stddev_parallel_csr'] * row['flops_parallel'] / row['time_parallel']
    })

# Per CSR SIMD
for _, row in df.iterrows():
    melted_data.append({
        'matrix_name': row['matrix_name'],
        'size_group': row['size_group'],
        'format': 'CSR SIMD',
        'threads': row['num_threads'],
        'flops': row['flops_parallel_simd'],
        'min_flops': row['flops_parallel_simd'] * (row['time_parallel_simd'] / row['min_value_csr_simd']),
        'stddev': row['stddev_parallel_simd'] * row['flops_parallel_simd'] / row['time_parallel_simd']
    })

melted_df = pd.DataFrame(melted_data)

# Definisci l'ordine desiderato dei gruppi
group_order = [
    "Gruppo 1: <10K",
    "Gruppo 2: 10K-100K",
    "Gruppo 3: 100K-500K",
    "Gruppo 4: 500K-1M",
    "Gruppo 5: 1M-2.5M",
    "Gruppo 6: >2.5M"
]

# Crea un box plot con seaborn
plt.figure(figsize=(14, 10))
ax = sns.boxplot(x='size_group', y='flops', hue='format',
                 data=melted_df, order=group_order,
                 palette=['#1f77b4', '#ff7f0e'])

# Aggiungi punti per i valori minimi
sns.stripplot(x='size_group', y='min_flops', hue='format',
              data=melted_df, order=group_order,
              palette=['#1f77b4', '#ff7f0e'], alpha=0.5,
              marker='D', size=7, jitter=0.2)

# Aggiungi linee di deviazione standard
for i, grp in enumerate(group_order):
    for j, fmt in enumerate(['CSR', 'CSR SIMD']):
        subset = melted_df[(melted_df['size_group'] == grp) & (melted_df['format'] == fmt)]
        if not subset.empty:
            mean_val = subset['flops'].mean()
            std_val = subset['stddev'].mean()  # Assuming stddev is already calculated properly

            # Calcola posizione x (offset per CSR vs CSR SIMD)
            x_pos = i + (-0.2 if j == 0 else 0.2)

            # Disegna le barre di errore per la deviazione standard
            plt.errorbar(x=x_pos, y=mean_val, yerr=std_val,
                         fmt='none', color='black', capsize=5, alpha=0.7)

# Personalizza il grafico
plt.yscale('log')  # Scala logaritmica per FLOPS se c'è grande variazione
plt.title('Performance SpMV: CSR vs CSR SIMD per Dimensione Matrice', fontsize=16)
plt.xlabel('Gruppi per Numero di Elementi Non-Nulli', fontsize=14)
plt.ylabel('FLOPS (operazioni in virgola mobile al secondo)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(title='Formato', fontsize=12, title_fontsize=13)

# Aggiungi annotazioni per chiarire i componenti del grafico
plt.figtext(0.15, 0.02,
            "Box: quartili (25%, mediana, 75%)\n"
            "Whiskers: range (min-max esclusi outlier)\n"
            "Diamanti: valori minimi osservati\n"
            "Barre verticali: deviazione standard",
            fontsize=12, ha='left')

plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.savefig('../result/spmv_performance_csr_base_simd_boxplot.png', dpi=300)
plt.show()