import numpy as np
import random
from scipy import sparse
import os
import datetime

M = 10
N = 10

class MTXGenerator:
    def __init__(self):
        self.matrix_types = {
            'general': 'general',
            'symmetric': 'symmetric',
            'pattern': 'pattern'
        }
        self.data_types = {
            'real': 'real',
            'integer': 'integer',
            'pattern': 'pattern'
        }

    def generate_header(self, matrix_type, data_type, rows, cols, nnz, title="", author="", date=None):
        """Genera l'intestazione per un file MTX"""
        if date is None:
            date = datetime.datetime.now().strftime("%Y")

        header = []
        header.append(f"%%MatrixMarket matrix coordinate {data_type} {matrix_type}")
        header.append("%-------------------------------------------------------------------------------")
        header.append(f"% Generated Matrix")
        header.append(f"% name: Generated/{title}")
        header.append(f"% date: {date}")
        header.append(f"% author: {author}")
        header.append(f"% kind: test matrix")
        header.append("%-------------------------------------------------------------------------------")
        header.append(f"{rows} {cols} {nnz}")

        return "\n".join(header)

    def generate_random_sparse_mtx(self, rows, cols, density=0.01, matrix_type='general',
                                   data_type='real', min_val=-1.0, max_val=1.0, filename=None,
                                   title="random_matrix", author="Generator"):
        """
        Genera una matrice sparsa casuale in formato MTX

        Args:
            rows: numero di righe
            cols: numero di colonne
            density: densità della matrice (frazione di elementi non zero)
            matrix_type: tipo di matrice ('general', 'symmetric', 'pattern')
            data_type: tipo di dati ('real', 'integer', 'pattern')
            min_val: valore minimo per elementi non zero (per 'real' e 'integer')
            max_val: valore massimo per elementi non zero (per 'real' e 'integer')
            filename: nome del file per salvare la matrice

        Returns:
            Matrice in formato MTX come stringa o salva su file se filename è fornito
        """
        # Calcola il numero di elementi non zero
        if matrix_type == 'symmetric' and rows == cols:
            # Per matrici simmetriche, generiamo solo la metà triangolare
            max_elements = (rows * (rows + 1)) // 2
            actual_nnz = int(density * max_elements)
        else:
            max_elements = rows * cols
            actual_nnz = int(density * max_elements)

        # Genera indici casuali per gli elementi non zero
        triplets = []


        if matrix_type == 'symmetric' and rows == cols:
            # Per matrici simmetriche, generiamo solo la metà triangolare superiore (o inferiore)
            indices = set()
            while len(indices) < actual_nnz:
                i = random.randint(0, rows-1)
                j = random.randint(0, cols-1)
                if i <= j and (i != 8 or j != 8):  # Evita di sovrascrivere la voce speciale (9,9) (indice 8 in base-0)
                    indices.add((i, j))

            for i, j in indices:
                if data_type == 'pattern':
                    triplets.append((i+1, j+1, None))  # Converti a base-1 per MTX
                elif data_type == 'integer':
                    triplets.append((i+1, j+1, random.randint(int(min_val), int(max_val))))
                else:  # 'real'
                    triplets.append((i+1, j+1, random.uniform(min_val, max_val)))

        else:
            # Per matrici generali
            indices = set()
            while len(indices) < actual_nnz:
                i = random.randint(0, rows-1)
                j = random.randint(0, cols-1)
                if (i != 8 or j != 8):  # Evita di sovrascrivere la voce speciale (9,9) (indice 8 in base-0)
                    indices.add((i, j))

            for i, j in indices:
                if data_type == 'pattern':
                    triplets.append((i+1, j+1, None))  # Converti a base-1 per MTX
                elif data_type == 'integer':
                    triplets.append((i+1, j+1, random.randint(int(min_val), int(max_val))))
                else:  # 'real'
                    triplets.append((i+1, j+1, random.uniform(min_val, max_val)))


        # Ordina per colonna e poi per riga
        triplets.sort(key=lambda x: (x[1], x[0]))  # Ordina per colonna (indice 1) e poi per riga (indice 0)

        # Crea il contenuto del file MTX
        header = self.generate_header(matrix_type, data_type, rows, cols, len(triplets),
                                      title=title, author=author)

        content = [header]
        for row, col, val in triplets:
            if data_type == 'pattern':
                # Per matrici pattern, non ci sono valori
                content.append(f"{row} {col}")
            elif data_type == 'integer':
                content.append(f"{row} {col} {val}")
            else:  # 'real'
                content.append(f"{row} {col} {val}")

        mtx_content = "\n".join(content)

        if filename:
            with open(filename, 'w') as f:
                f.write(mtx_content)
            print(f"Matrice MTX salvata in {filename}")

        return mtx_content

    def mtx_to_csr(self, mtx_content):
        """
        Converte una matrice in formato MTX in formato CSR

        Args:
            mtx_content: Contenuto del file MTX come stringa

        Returns:
            Matrice in formato CSR (scipy.sparse.csr_matrix)
        """
        lines = mtx_content.strip().split('\n')

        # Estrai informazioni sul tipo della matrice dal primo commento
        header = lines[0].strip()
        is_symmetric = "symmetric" in header.lower()
        is_pattern = "pattern" in header.lower()

        # Trova la prima riga di dati (che contiene le dimensioni)
        data_start = 0
        for i, line in enumerate(lines):
            if not line.startswith('%'):
                data_start = i
                break

        # Prima riga contiene le dimensioni e il numero di elementi non zero
        dimensions = lines[data_start].split()
        rows = int(dimensions[0])
        cols = int(dimensions[1])

        # Estrai triplets (riga, colonna, valore)
        row_indices = []
        col_indices = []
        values = []

        for i in range(data_start + 1, len(lines)):
            if lines[i].strip():  # Ignora righe vuote
                parts = lines[i].split()
                if len(parts) >= 2:
                    # Converti a indici base-0 (dal formato MTX base-1)
                    row = int(parts[0]) - 1
                    col = int(parts[1]) - 1

                    if is_pattern:
                        val = 1  # Per matrici pattern, tutti i valori sono 1
                    else:
                        val = float(parts[2]) if len(parts) > 2 else 1

                    row_indices.append(row)
                    col_indices.append(col)
                    values.append(val)

                    # Se la matrice è simmetrica, aggiungi anche l'elemento simmetrico
                    # (ma solo se non è sulla diagonale)
                    if is_symmetric and row != col:
                        row_indices.append(col)
                        col_indices.append(row)
                        values.append(val)

        # Crea la matrice CSR
        csr_matrix = sparse.csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(rows, cols)
        )

        return csr_matrix

    def print_csr_info(self, csr_matrix):
        """
        Stampa le informazioni di una matrice CSR
        """
        print(f"Dimensioni: {csr_matrix.shape}")
        print(f"Elementi non-zero: {csr_matrix.nnz}")
        print("Indici delle righe (row_ptr):", csr_matrix.indptr)
        print("Indici delle colonne (col_idx):", csr_matrix.indices)
        print("Valori:", csr_matrix.data)

    def save_csr_to_c_format(self, csr_matrix, filename):
        """
        Salva una matrice CSR in un formato C-friendly
        """
        with open(filename, 'w') as f:
            f.write("// CSR Matrix representation\n")
            f.write(f"int M = {csr_matrix.shape[0]};\n")
            f.write(f"int N = {csr_matrix.shape[1]};\n")
            f.write(f"int nz = {csr_matrix.nnz};\n")

            f.write("\n// Row pointers\n")
            f.write("int row_ptr[] = {")
            f.write(", ".join(map(str, csr_matrix.indptr)))
            f.write("};\n")

            f.write("\n// Column indices\n")
            f.write("int col_idx[] = {")
            f.write(", ".join(map(str, csr_matrix.indices)))
            f.write("};\n")

            f.write("\n// Values\n")
            f.write("double values[] = {")
            f.write(", ".join(map(str, csr_matrix.data)))
            f.write("};\n")

        print(f"Matrice CSR salvata in formato C in {filename}")


if __name__ == "__main__":
    generator = MTXGenerator()

    # Genera una matrice general con ordinamento personalizzato
    general_mtx = generator.generate_random_sparse_mtx(
        rows=M,
        cols=N,
        density=0.05,
        matrix_type='general',
        data_type='real',
        filename="../matrix_generated/general_matrix.mtx",
        title="general_test",
    )

    # Genera una matrice simmetrica
    symmetric_mtx = generator.generate_random_sparse_mtx(
        rows=M,
        cols=N,
        density=0.05,
        matrix_type='symmetric',
        data_type='real',
        filename="../matrix_generated/symmetric_matrix.mtx",
        title="symmetric_test",
    )

    # Genera una matrice pattern
    pattern_mtx = generator.generate_random_sparse_mtx(
        rows=M,
        cols=N,
        density=0.05,
        matrix_type='general',
        data_type='pattern',
        filename="../matrix_generated/pattern_matrix.mtx",
        title="pattern_test",
    )

    # Converti in CSR e stampa info
    csr_matrix = generator.mtx_to_csr(general_mtx)
    print("Matrice generale:")
    generator.print_csr_info(csr_matrix)
    x = np.array([1,1,1,1,5,6,7,8,9,10])
    y = csr_matrix @ x
    print("Vettore risultante SpMV:", np.array2string(y, precision=3))

    # Converti in CSR e stampa info
    csr_matrix = generator.mtx_to_csr(pattern_mtx)
    print("Matrice pattern:")
    generator.print_csr_info(csr_matrix)
    x = np.array([1,1,1,1,5,6,7,8,9,10])
    y = csr_matrix @ x
    print("Vettore risultante SpMV:", np.array2string(y, precision=3))

    # Converti in CSR e stampa info
    csr_matrix = generator.mtx_to_csr(symmetric_mtx)
    print("Matrice simmetrica: ")
    generator.print_csr_info(csr_matrix)
    x = np.array([1,1,1,1,5,6,7,8,9,10])
    y = csr_matrix @ x
    print("Vettore risultante SpMV:", np.array2string(y, precision=3))

