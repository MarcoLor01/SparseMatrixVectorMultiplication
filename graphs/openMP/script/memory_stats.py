import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Funzione per leggere il file CSV e analizzare i dati per categoria
def analyze_memory_by_category(csv_path):
    # Verifica se il file esiste
    if not os.path.exists(csv_path):
        print(f"Il file {csv_path} non esiste.")
        return

    # Leggi il file CSV
    df = pd.read_csv(csv_path)

    # Definisci le categorie in base al numero di elementi non-zero
    categories = [
        ('< 10K', 0, 10_000),
        ('10K-100K', 10_000, 100_000),
        ('100K-500K', 100_000, 500_000),
        ('500K-1M', 500_000, 1_000_000),
        ('1M-2.5M', 1_000_000, 2_500_000),
        ('>= 2.5M', 2_500_000, float('inf'))
    ]

    # Crea una colonna di categoria
    def assign_category(nz):
        for cat_name, min_val, max_val in categories:
            if min_val <= nz < max_val:
                return cat_name
        return None

    df['Category'] = df['Non-Zero Elements'].apply(assign_category)

    # Statistiche per categoria
    cat_stats = df.groupby('Category').agg({
        'Memory Size (MB)': ['mean', 'min', 'max', 'count'],
        'Non-Zero Elements': ['mean', 'min', 'max']
    }).reset_index()

    # Flatten the column names
    cat_stats.columns = ['_'.join(col).strip('_') for col in cat_stats.columns.values]

    # L3 cache size in MB (esempio tipico, modificare in base al proprio processore)
    l3_cache_size = 28.83584

    # Preparazione per il grafico
    categories = cat_stats['Category']
    avg_memory = cat_stats['Memory Size (MB)_mean']

    # Stampa le statistiche
    print("\n=== STATISTICHE DI MEMORIA PER CATEGORIA ===")
    for i, cat in enumerate(categories):
        print(f"\nCategoria: {cat}")
        print(f"Numero di matrici: {cat_stats['Memory Size (MB)_count'][i]}")
        print(f"Elementi non-zero: min={cat_stats['Non-Zero Elements_min'][i]:.0f}, "
              f"max={cat_stats['Non-Zero Elements_max'][i]:.0f}, "
              f"media={cat_stats['Non-Zero Elements_mean'][i]:.0f}")
        print(f"Memoria (MB): min={cat_stats['Memory Size (MB)_min'][i]:.2f}, "
              f"max={cat_stats['Memory Size (MB)_max'][i]:.2f}, "
              f"media={cat_stats['Memory Size (MB)_mean'][i]:.2f}")

        # Confronto con cache L3
        ratio_to_l3 = cat_stats['Memory Size (MB)_mean'][i] / l3_cache_size
        print(f"Rapporto con L3 cache ({l3_cache_size} MB): {ratio_to_l3:.2f}x")
        if ratio_to_l3 > 1:
            print(f"ATTENZIONE: La memoria media eccede la dimensione della cache L3 di {ratio_to_l3:.2f} volte")
        else:
            print(f"OK: La memoria media è {ratio_to_l3:.2f} volte la dimensione della cache L3")

    # Crea il grafico
    plt.figure(figsize=(12, 8))

    # Grafico a barre per la memoria media
    bars = plt.bar(categories, avg_memory, color='steelblue', alpha=0.7)

    # Aggiungi linea orizzontale per la dimensione della cache L3
    plt.axhline(y=l3_cache_size, color='red', linestyle='--',
                label=f'L3 Cache Size ({l3_cache_size} MB)')

    # Aggiungi etichette con il valore sopra ogni barra
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f} MB', ha='center', va='bottom')

    plt.title('Memoria Media per Categoria vs. Dimensione Cache L3', fontsize=14)
    plt.xlabel('Categoria (elementi non-zero)', fontsize=12)
    plt.ylabel('Memoria Media (MB)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    # Salva il grafico
    output_path = '../../../result/memory_analysis_hll.png'
    plt.savefig(output_path)
    print(f"\nGrafico salvato in {output_path}")

    plt.tight_layout()
    plt.show()

    return df, cat_stats

if __name__ == "__main__":
    csv_path = "../../../result/memory_hll.csv"

    # Esegui l'analisi
    df, stats = analyze_memory_by_category(csv_path)

    print("\n=== ANALISI CONCLUSIONI ===")
    print("L'analisi mostra se le matrici di diverse categorie superano la dimensione della cache L3.")
    print("Quando una matrice è più grande della cache L3, si verificano più cache miss durante SpMV,")
    print("causando accessi più frequenti alla memoria principale, che è molto più lenta.")
    print("\nLe matrici che superano significativamente la dimensione della L3 (>= 2.5M elementi)")
    print("mostrano prestazioni peggiori perché:")
    print("1. Generano più cache miss")
    print("2. Richiedono più trasferimenti dalla memoria principale")
    print("3. Creano contesa tra i thread per lo spazio limitato in cache")

    # Aggiungi l'analisi del working set
    print("\n=== ANALISI DEL WORKING SET ===")
    print("Nota che questa analisi considera solo la dimensione della matrice CSR.")
    print("Il working set completo durante SpMV include anche:")
    print("1. Il vettore di input (solitamente di dimensione N)")
    print("2. Il vettore di output (solitamente di dimensione M)")
    print("Questi aumentano ulteriormente l'occupazione di memoria effettiva durante il calcolo.")