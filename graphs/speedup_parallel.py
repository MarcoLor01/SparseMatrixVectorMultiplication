import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carica il file CSV
df = pd.read_csv("../result/spmv_results.csv")

# Debug: Stampa le righe con valori problematici in time_parallel
problematic_rows = df[(df['time_parallel'] == 0) | (df['time_parallel'].isna())]
if len(problematic_rows) > 0:
    print("RIGHE PROBLEMATICHE (time_parallel = 0 o NaN):")
    print(problematic_rows)
    print(f"\nTotale righe problematiche: {len(problematic_rows)} su {len(df)} ({len(problematic_rows)/len(df)*100:.2f}%)")

# Rimuovi righe con time_parallel problematico
df_filtered = df[(df['time_parallel'] > 0) & (~df['time_parallel'].isna())].copy()

# Se non ci sono dati validi, termina
if len(df_filtered) == 0:
    print("Attenzione: Nessun dato valido dopo il filtraggio!")
    exit(1)

# Calcolo dello Speedup
df_filtered['speedup_parallel'] = df_filtered['time_serial'] / df_filtered['time_parallel']

# Definizione delle categorie basate sui nonzeros
nonzero_categories = [
    (0, 10000, '< 10K'),         # < 10K
    (10000, 100000, '10K < NZ <= 100K'),          # 10K - 100K
    (100000, 500000, '100K < NZ <= 500K'),           # 100K - 500K
    (500000, 2000000, '500K < NZ <= 2M'),   # 500K - 1M
    (2000000, 10000000, '2M < NZ <= 10M'), # 2M - 10M
    (10000000, float('inf'), '10M < NZ'),        # 10M +
]

def categorize_nonzeros(nz):
    for min_nz, max_nz, category in nonzero_categories:
        if min_nz <= nz < max_nz:
            return category
    return 'enormi'

df_filtered['nz_category'] = df_filtered['nonzeros'].apply(categorize_nonzeros)

# Debug: Mostra il numero di matrici per categoria
category_counts = df_filtered['nz_category'].value_counts()
print("\nDistribuzione per categoria:")
for category, count in category_counts.items():
    print(f"  {category}: {count} righe")

# Creazione del grafico
plt.figure(figsize=(12, 8))

# Colori e marker per le diverse categorie
colors = {
    '< 10K': 'blue',
    '10K < NZ <= 100K': 'green',
    '100K < NZ <= 500K': 'red',
    '500K < NZ <= 2M': 'orange',
    '2M < NZ <= 10M': 'black',
    '10M < NZ': 'purple',
}

markers = {
    '< 10K': 'o',
    '10K < NZ <= 100K': 's',
    '100K < NZ <= 500K': '^',
    '500K < NZ <= 2M': 'D',
    '2M < NZ <= 10M': 'p',
    '10M < NZ': '*',
}

# Per ogni categoria di matrice
for category in sorted(df_filtered['nz_category'].unique()):
    category_data = df_filtered[df_filtered['nz_category'] == category]
    grouped_data = category_data.groupby('num_threads')['speedup_parallel'].mean().reset_index()
    grouped_data = grouped_data.sort_values('num_threads')

    plt.plot(grouped_data['num_threads'], grouped_data['speedup_parallel'],
             marker=markers.get(category, 'o'),
             linestyle='-',
             color=colors.get(category, 'black'),
             label=f'Matrici {category} (max {grouped_data["speedup_parallel"].max():.2f}x)')

plt.title('Speedup Parallelizzato in base al numero di thread e dimensione delle matrici')
plt.xlabel('Numero di thread')
plt.ylabel('Speedup')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title="Categoria di matrici per nonzeros", loc='upper left')
plt.yscale('linear')
plt.xticks(sorted(df_filtered['num_threads'].unique()))

# Salva il grafico
try:
    plt.savefig("spmv_speedup_by_matrix_size.png", dpi=300, bbox_inches='tight')
    print("\nGrafico salvato come 'spmv_speedup_by_matrix_size.png'")
except Exception as e:
    print(f"\nErrore nel salvare il grafico: {e}")

plt.show()
