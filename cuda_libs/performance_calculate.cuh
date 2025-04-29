//
// Created by HP on 09/03/2025.
//

#define INITIAL_CAPACITY 100
#ifndef PERFORMANCE_CALCULATE_CUH
#define PERFORMANCE_CALCULATE_CUH
#include <stddef.h>

typedef struct {
    double sum;         // Per calcolare la media
    double* values;     // Array di valori per calcolare varianza/deviazione standard
    double relative_error;       // Errore relativo
    double absolute_error;       // Errore assoluto
    int count;          // Contatore delle misurazioni
    int capacity;       // Capacità dell'array values
} MetricStats;

typedef enum {
    SERIAL_TIME,
    ROW_CSR_TIME,
    WARP_CSR_TIME,
    ROW_HLL_TIME,
    WARP_HLL_TIME,
    WARP_SHARED_MEMORY_CSR_TIME,
    SERIAL_HLL_TIME,
    WARP_SHARED_MEMORY_HLL_TIME,
    NUM_METRICS,
} MediumPerformanceMetric;

typedef struct DifferenceMetrics {
    double mean_abs_err;
    double mean_rel_err;
    int iterations;
} DiffMetrics;

typedef struct performance_metrics {
    double time;
    double flops;
} PerformanceMetrics;

double get_metric_value(MediumPerformanceMetric type);
void update_medium_metric(MediumPerformanceMetric type, double value);
double calculate_flops(int nz, double time);
void reset_medium_time_metrics();
void print_flops(double flops);
void cleanup_metrics();
void initialize_metrics();
DifferenceMetrics computeDifferenceMetrics(
    const double* ref,
    const double* res,
    int n,
    double abs_tol = 1e-5,
    double rel_tol = 1e-4,
    bool print_summary = true
);
double get_relative_error(const MediumPerformanceMetric type);
double get_absolute_error(const MediumPerformanceMetric type);
void accumulateErrors(const DiffMetrics* iteration_metrics, const MediumPerformanceMetric type);
DiffMetrics computeAverageErrors(const MediumPerformanceMetric type);
#endif //PERFORMANCE_CALCULATE_H
