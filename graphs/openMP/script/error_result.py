import pandas as pd
import matplotlib.pyplot as plt
import os

# Carica il CSV
df = pd.read_csv('../../../result/spmv_results_definitive.csv')

# Etichette leggibili
labels = {
    'error_csr': 'Errore CSR',
    'error_csr_simd': 'Errore CSR + SIMD',
    'error_hll': 'Errore HLL',
    'error_hll_simd': 'Errore HLL + SIMD'
}

# --- GRAFICO CSR ---
plt.figure(figsize=(14, 6))
for threads in sorted(df['num_threads'].unique()):
    df_filtered = df[df['num_threads'] == threads]
    plt.plot(df_filtered['matrix_name'], df_filtered['error_csr'], marker='o', label=f'CSR - {threads} thread')
    plt.plot(df_filtered['matrix_name'], df_filtered['error_csr_simd'], marker='x', linestyle='--', label=f'CSR+SIMD - {threads} thread')

plt.title('Errori CSR e CSR+SIMD al variare dei thread')
plt.xlabel('Nome Matrice')
plt.ylabel('Errore')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'../result/error_csr+csr_SIMD.png')
plt.close()

# --- GRAFICO HLL ---
plt.figure(figsize=(14, 6))
for threads in sorted(df['num_threads'].unique()):
    df_filtered = df[df['num_threads'] == threads]
    plt.plot(df_filtered['matrix_name'], df_filtered['error_hll'], marker='o', label=f'HLL - {threads} thread')
    plt.plot(df_filtered['matrix_name'], df_filtered['error_hll_simd'], marker='x', linestyle='--', label=f'HLL+SIMD - {threads} thread')

plt.title('Errori HLL e HLL+SIMD al variare dei thread')
plt.xlabel('Nome Matrice')
plt.ylabel('Errore')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'../result/error_hll+hll_SIMD.png')
plt.close()

