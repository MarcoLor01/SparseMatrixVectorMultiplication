#include <stdio.h>
#include <stdlib.h>
#include "matrix_parser.h"

#include <stdbool.h>
#include <string.h>

void init_pre_matrix(PreMatrix *mat) {
    mat->M = 0;
    mat->N = 0;
    mat->nz = 0;
    mat->I = NULL;
    mat->J = NULL;
    mat->val = NULL;
}

void free_pre_matrix(PreMatrix *mat) {
    free(mat->I);
    free(mat->J);
    free(mat->val);
    init_pre_matrix(mat);
}

int read_matrix_market(const char *filename, PreMatrix *mat) {
    FILE *f;

    if ((f = fopen(filename, "r")) == NULL) {
        printf("Errore nell'apertura del file\n");
        return -1;
    }

    if (mm_read_banner(f, &mat->type) != 0) {
        printf("Formato Matrix Market non riconosciuto.\n");
        fclose(f);
        return -1;
    }

    if (!mm_is_matrix(mat->type) || !mm_is_sparse(mat->type)) {
        printf("Sono supportare solo matrici sparse.\n");
        fclose(f);
        return -1;
    }

    int original_nz;

    //Leggo parametri matrice
    if (mm_read_mtx_crd_size(f, &mat->M, &mat->N, &original_nz) != 0) {
        fclose(f);
        return -1;
    }

    const int is_symmetric = mm_is_symmetric(mat->type);
    const int max_nz = is_symmetric ? original_nz * 2 : original_nz;

    int *temp_I = malloc(max_nz * sizeof(int));
    int *temp_J = malloc(max_nz * sizeof(int));
    double *temp_val = malloc(max_nz * sizeof(double));
    if (!temp_I || !temp_J || !temp_val) {
        free(temp_I);
        free(temp_J);
        free(temp_val);
        fclose(f);
        printf("Errore nell'allocazione della memoria\n");
        return -1;
    }

    int actual_nz = 0;
    for (int i = 0; i < original_nz; i++) {
        if (mm_is_pattern(mat->type)) {
            int result = fscanf(f, "%d %d", &temp_I[actual_nz], &temp_J[actual_nz]);
            if (result != 2) {
                printf("Errore di lettura pattern alla riga %d: letti %d valori invece di 2\n", i+1, result);
                char buffer[100];
                fgets(buffer, sizeof(buffer), f);
                printf("Contenuto della riga problematica: %s\n", buffer);
                free(temp_I);
                free(temp_J);
                free(temp_val);
                fclose(f);
                return -1;
            }
            temp_val[actual_nz] = 1.0;
        } else {

            int result = fscanf(f, "%d %d %lf", &temp_I[actual_nz], &temp_J[actual_nz], &temp_val[actual_nz]);
            if (result != 3) {
                printf("Errore di lettura alla riga %d: letti %d valori invece di 3\n", i+1, result);
                char buffer[100];
                fgets(buffer, sizeof(buffer), f);
                printf("Contenuto della riga problematica: %s\n", buffer);
                free(temp_I);
                free(temp_J);
                free(temp_val);
                fclose(f);
                return -1;
            }
        }

        // Converti da 1-based a 0-based
        temp_I[actual_nz]--;
        temp_J[actual_nz]--;

        if (temp_I[actual_nz] < 0 || temp_I[actual_nz] >= mat->M ||
            temp_J[actual_nz] < 0 || temp_J[actual_nz] >= mat->N) {
            printf("Errore: Indice fuori range (%d,%d) per matrice %dx%d\n",
                   temp_I[actual_nz] + 1, temp_J[actual_nz] + 1, mat->M, mat->N);
            free(temp_I);
            free(temp_J);
            free(temp_val);
            fclose(f);
            return -1;
        }
        actual_nz++;

        //Simmetria e non valore sulla diagonale
        if (is_symmetric && temp_I[actual_nz-1] != temp_J[actual_nz-1]) {
            temp_I[actual_nz] = temp_J[actual_nz-1]; //Scambio riga e colonna
            temp_J[actual_nz] = temp_I[actual_nz-1];
            temp_val[actual_nz] = temp_val[actual_nz-1]; //Valore simmetrico
            actual_nz++;
        }
    }

    fclose(f);

    mat->nz = actual_nz;
    mat->I = (int *)malloc(mat->nz * sizeof(int));
    mat->J = (int *)malloc(mat->nz * sizeof(int));
    mat->val = (double *)malloc(mat->nz * sizeof(double));

    if (!mat->I || !mat->J || !mat->val) {
        printf("Errore di allocazione memoria\n");
        free(temp_I);
        free(temp_J);
        free(temp_val);
        free(mat->I);
        free(mat->J);
        free(mat->val);
        return -1;
    }

    memcpy(mat->I, temp_I, mat->nz * sizeof(int));
    memcpy(mat->J, temp_J, mat->nz * sizeof(int));
    memcpy(mat->val, temp_val, mat->nz * sizeof(double));

    free(temp_I);
    free(temp_J);
    free(temp_val);
    return 0;
}

void print_pre_matrix(const PreMatrix *mat, bool const full_print) {
    printf("Dimensioni matrice: %d x %d\n", mat->M, mat->N);
    printf("Numero di elementi non-zero: %d\n", mat->nz);
    printf("Matrix type: %s\n", mm_typecode_to_str(mat->type));
    if (mat->M <= 30) {
        if (full_print == true){
            printf("Indice righe (I): ");
            for (int i = 0; i < mat->nz; i++) {
                printf("%d ", mat->I[i]);
            }
            printf("\n");

            printf("Indice colonne (J): ");
            for (int i = 0; i < mat->nz; i++) {
                printf("%d ", mat->J[i]);
            }
            printf("\n");

            printf("Valori: ");
            for (int i = 0; i < mat->nz; i++) {
                printf("%f", mat->val[i]);
            }
            printf("\n");
        }
    }
}