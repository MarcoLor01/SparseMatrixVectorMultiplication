import pandas as pd
import matplotlib.pyplot as plt
import os

# Crea directory per i grafici se non esiste
output_dir = "spmv_performance_charts"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Carica il file CSV
df = pd.read_csv("../result/spmv_results.csv")

# Debug: Stampa le righe problematiche
problematic_rows = df[(df['flops_parallel'] <= 0) | (df['flops_parallel'].isna()) |
                      (df['flops_parallel_unrolling'] <= 0) | (df['flops_parallel_unrolling'].isna())]

if not problematic_rows.empty:
    print("RIGHE PROBLEMATICHE (flops <= 0 o NaN):")
    print(problematic_rows)
    print(f"\nTotale righe problematiche: {len(problematic_rows)} su {len(df)} ({len(problematic_rows)/len(df)*100:.2f}%)")

# Filtra dati validi
df_filtered = df[(df['flops_parallel'] > 0) & (~df['flops_parallel'].isna()) &
                 (df['flops_parallel_unrolling'] > 0) & (~df['flops_parallel_unrolling'].isna())].copy()

if df_filtered.empty:
    print("Errore: Nessun dato valido dopo il filtraggio.")
    exit(1)

# Converti FLOPS in MEGAFLOPS
df_filtered['flops_parallel'] /= 1e6
df_filtered['flops_parallel_unrolling'] /= 1e6

# Definizione categorie basate su nonzeros
nonzero_categories = [
    (0, 10000, '< 10K'),
    (10000, 100000, '10K-100K'),
    (100000, 500000, '100K-500K'),
    (500000, 2000000, '500K-2M'),
    (2000000, 10000000, '2M-10M'),
    (10000000, float('inf'), '> 10M')
]

def categorize_nonzeros(nz):
    for min_nz, max_nz, category in nonzero_categories:
        if min_nz <= nz < max_nz:
            return category
    return 'Altro'

df_filtered['nz_category'] = df_filtered['nonzeros'].apply(categorize_nonzeros)

# Debug: Mostra la distribuzione
print("\nDistribuzione per categoria:")
print(df_filtered['nz_category'].value_counts())

# Imposta colori e marker
colors = {
    '< 10K': '#1f77b4',
    '10K-100K': '#ff7f0e',
    '100K-500K': '#2ca02c',
    '500K-2M': '#d62728',
    '2M-10M': '#9467bd',
    '> 10M': '#8c564b'
}

# Calcola il miglioramento percentuale
df_filtered['improvement'] = ((df_filtered['flops_parallel_unrolling'] - df_filtered['flops_parallel']) /
                              df_filtered['flops_parallel'] * 100)

# 1. GRAFICO COMPLESSIVO
plt.figure(figsize=(14, 10))

for category in sorted(df_filtered['nz_category'].unique()):
    category_data = df_filtered[df_filtered['nz_category'] == category]

    # Dati standard
    grouped_standard = category_data.groupby('num_threads')['flops_parallel'].mean().reset_index()
    plt.plot(grouped_standard['num_threads'], grouped_standard['flops_parallel'],
             marker='o', linestyle='-', color=colors.get(category, 'black'),
             label=f'Standard {category}')

    # Dati unrolling
    grouped_unrolling = category_data.groupby('num_threads')['flops_parallel_unrolling'].mean().reset_index()
    plt.plot(grouped_unrolling['num_threads'], grouped_unrolling['flops_parallel_unrolling'],
             marker='s', linestyle='--', color=colors.get(category, 'gray'),
             label=f'Unrolling {category}')

plt.title('Confronto Prestazioni SpMV: Standard vs Unrolling', fontsize=16)
plt.xlabel('Numero di Thread', fontsize=14)
plt.ylabel('MEGAFLOPS', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title="Categoria di matrici", loc='upper left', fontsize=12)
plt.xticks(sorted(df_filtered['num_threads'].unique()))
plt.tight_layout()
plt.savefig(f"{output_dir}/spmv_all_categories_comparison.png", dpi=300)
print(f"Grafico complessivo salvato in {output_dir}/spmv_all_categories_comparison.png")

# 2. GRAFICI PER CATEGORIA
for category in sorted(df_filtered['nz_category'].unique()):
    plt.figure(figsize=(12, 8))

    category_data = df_filtered[df_filtered['nz_category'] == category]

    # Dati standard
    grouped_standard = category_data.groupby('num_threads')['flops_parallel'].mean().reset_index()
    plt.plot(grouped_standard['num_threads'], grouped_standard['flops_parallel'],
             marker='o', markersize=10, linestyle='-', linewidth=2, color='#1f77b4',
             label='Standard')

    # Dati unrolling
    grouped_unrolling = category_data.groupby('num_threads')['flops_parallel_unrolling'].mean().reset_index()
    plt.plot(grouped_unrolling['num_threads'], grouped_unrolling['flops_parallel_unrolling'],
             marker='s', markersize=10, linestyle='--', linewidth=2, color='#ff7f0e',
             label='Unrolling')

    # Etichette miglioramento
    for i in range(len(grouped_standard)):
        thread = grouped_standard['num_threads'].iloc[i]
        std_val = grouped_standard['flops_parallel'].iloc[i]
        unr_val = grouped_unrolling['flops_parallel_unrolling'].iloc[i]
        improvement = ((unr_val - std_val) / std_val * 100)

        # Posiziona l'etichetta sopra il punto più alto
        y_pos = max(std_val, unr_val) + (max(grouped_unrolling['flops_parallel_unrolling'].max(),
                                             grouped_standard['flops_parallel'].max()) * 0.03)

        if improvement > 0:
            color = 'green'
            label = f"+{improvement:.1f}%"
        else:
            color = 'red'
            label = f"{improvement:.1f}%"

        plt.annotate(label, (thread, y_pos), color=color,
                     fontweight='bold', ha='center', va='bottom')

    plt.title(f'Confronto Prestazioni SpMV - Categoria {category}', fontsize=16)
    plt.xlabel('Numero di Thread', fontsize=14)
    plt.ylabel('MEGAFLOPS', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Versione", loc='upper left', fontsize=12)
    plt.xticks(sorted(category_data['num_threads'].unique()))

    # Aggiungi informazioni sulla categoria
    num_matrices = len(category_data['matrix_name'].unique())
    plt.figtext(0.5, 0.01, f"Basato su {num_matrices} matrici uniche",
                ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/spmv_category_{category.replace(' ', '_').replace('<', 'lt').replace('>', 'gt')}.png", dpi=300)
    print(f"Grafico categoria {category} salvato")

# 3. GRAFICO DI MIGLIORAMENTO PERCENTUALE
plt.figure(figsize=(14, 10))

for category in sorted(df_filtered['nz_category'].unique()):
    category_data = df_filtered[df_filtered['nz_category'] == category]
    grouped_improvement = category_data.groupby('num_threads')['improvement'].mean().reset_index()

    plt.plot(grouped_improvement['num_threads'], grouped_improvement['improvement'],
             marker='D', markersize=8, linestyle='-', linewidth=2, color=colors.get(category, 'black'),
             label=f'Categoria {category}')

plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title('Miglioramento Percentuale con Unrolling per Categoria', fontsize=16)
plt.xlabel('Numero di Thread', fontsize=14)
plt.ylabel('Miglioramento %', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title="Categoria di matrici", loc='best', fontsize=12)
plt.xticks(sorted(df_filtered['num_threads'].unique()))
plt.tight_layout()
plt.savefig(f"{output_dir}/spmv_improvement_percentage.png", dpi=300)
print(f"Grafico miglioramento percentuale salvato in {output_dir}/spmv_improvement_percentage.png")

# 4. GRAFICO DI SPEEDUP
plt.figure(figsize=(14, 10))

# Calcola speedup per ogni categoria e versione
for category in sorted(df_filtered['nz_category'].unique()):
    category_data = df_filtered[df_filtered['nz_category'] == category]

    # Trova i valori per thread=1 che saranno usati come baseline
    baseline_standard = category_data[category_data['num_threads'] == 1]['flops_parallel'].mean()
    baseline_unrolling = category_data[category_data['num_threads'] == 1]['flops_parallel_unrolling'].mean()

    # Calcola speedup standard
    grouped_standard = category_data.groupby('num_threads')['flops_parallel'].mean().reset_index()
    grouped_standard['speedup'] = grouped_standard['flops_parallel'] / baseline_standard

    # Calcola speedup unrolling
    grouped_unrolling = category_data.groupby('num_threads')['flops_parallel_unrolling'].mean().reset_index()
    grouped_unrolling['speedup'] = grouped_unrolling['flops_parallel_unrolling'] / baseline_unrolling

    # Disegna linee di speedup
    plt.plot(grouped_standard['num_threads'], grouped_standard['speedup'],
             marker='o', linestyle='-', color=colors.get(category, 'black'),
             label=f'Standard {category}')

    plt.plot(grouped_unrolling['num_threads'], grouped_unrolling['speedup'],
             marker='s', linestyle='--', color=colors.get(category, 'gray'),
             label=f'Unrolling {category}')

# Aggiungi linea di speedup ideale
max_threads = df_filtered['num_threads'].max()
plt.plot([1, max_threads], [1, max_threads], 'k:', label='Speedup Ideale')

plt.title('Speedup per Categoria e Versione', fontsize=16)
plt.xlabel('Numero di Thread', fontsize=14)
plt.ylabel('Speedup', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title="Versione per Categoria", loc='upper left', fontsize=10)
plt.xticks(sorted(df_filtered['num_threads'].unique()))
plt.tight_layout()
plt.savefig(f"{output_dir}/spmv_speedup_comparison.png", dpi=300)
print(f"Grafico speedup salvato in {output_dir}/spmv_speedup_comparison.png")

print("\nAnalisi completata. Tutti i grafici sono stati salvati nella directory", output_dir)