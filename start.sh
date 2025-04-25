#!/bin/bash

# Directory del progetto (modifica se necessario)
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="${PROJECT_DIR}/build"

# Funzione di uso
usage() {
    echo "Uso: $0 [openmp|cuda]"
    echo "  openmp: Compila versione OpenMP"
    echo "  cuda: Compila versione CUDA"
    exit 1
}

# Controllo argomenti
if [ $# -ne 1 ]; then
    usage
fi

# Carica moduli necessari
module load cuda

# Elimina e ricrea la directory di build
rm -rf "${BUILD_DIR}"
echo "Creazione directory: ${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Compila in base all'argomento
case "$1" in
    openmp)
        cmake -DENABLE_OPENMP=ON -DENABLE_CUDA=OFF "${PROJECT_DIR}"
        make
        ./SpMV_OpenMP
        ;;
    cuda)
        cmake -DENABLE_CUDA=ON -DENABLE_OPENMP=OFF "${PROJECT_DIR}"
        make
        ./SpMV_CUDA
        ;;
    *)
        usage
        ;;
esac