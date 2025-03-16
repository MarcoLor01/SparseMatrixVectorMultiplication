import pandas as pd
import matplotlib.pyplot as plt

# Carica il file CSV
df = pd.read_csv("../result/spmv_results.csv")

# Filtra i dati problematici
df_filtered = df[(df['efficiency_parallel'] > 0) & (~df['efficiency_parallel'].isna())].copy()

# Controllo dopo il filtraggio
if df_filtered.empty:
    print("Errore: Nessun dato valido trovato dopo il filtraggio.")
    exit(1)

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

# Stampa i valori di efficiency_parallel divisi per categoria e num_threads
print("\nEfficienza Parallelizzata per categoria e numero di thread:\n")
for category in sorted(df_filtered['nz_category'].unique()):
    category_data = df_filtered[df_filtered['nz_category'] == category]
    print(f"\nCategoria: {category}")
    for num_threads in sorted(category_data['num_threads'].unique()):
        avg_efficiency = category_data[category_data['num_threads'] == num_threads]['efficiency_parallel'].mean()
        print(f"Threads: {num_threads}, Efficienza media: {avg_efficiency:.4f}")

# Genera un grafico per visualizzare l'efficienza
groups = df_filtered.groupby(['nz_category', 'num_threads'])['efficiency_parallel'].mean().reset_index()
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

for category in sorted(df_filtered['nz_category'].unique()):
    subset = groups[groups['nz_category'] == category]
    plt.plot(subset['num_threads'], subset['efficiency_parallel'],
             marker=markers[category], linestyle='-', color=colors[category],
             label=f'Matrici {category}')

plt.title('Efficienza Parallelizzata in base al numero di thread e dimensione delle matrici')
plt.xlabel('Numero di thread')
plt.ylabel('Efficienza')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title="Categoria di matrici", loc='upper right')
plt.xticks(sorted(df_filtered['num_threads'].unique()))

try:
    plt.savefig("efficiency_by_matrix_size.png", dpi=300, bbox_inches='tight')
    print("\nGrafico salvato come 'efficiency_by_matrix_size.png'")
except Exception as e:
    print(f"\nErrore nel salvare il grafico: {e}")

plt.show()