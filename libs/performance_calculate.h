//
// Created by HP on 09/03/2025.
//

#ifndef PERFORMANCE_CALCULATE_H
#define PERFORMANCE_CALCULATE_H
#include <stddef.h>
#include <stdbool.h>

#define INITIAL_CAPACITY 100

typedef struct {
    double sum;         // Per calcolare la media
    double min;         // Valore minimo
    double max;         // Valore massimo
    double* values;     // Array di valori per calcolare varianza/deviazione standard
    double relative_error;       // Errore relativo
    double absolute_error;       // Errore assoluto
    int count;          // Contatore delle misurazioni
    int capacity;       // Capacità dell'array values
} MetricStats;

typedef enum {
    SERIAL_TIME,
    PARALLEL_CSR_TIME,
    PARALLEL_SIMD_CSR_TIME,
    PARALLEL_HLL_TIME,
    PARALLEL_HLL_SIMD_TIME,
    SERIAL_HLL_TIME,
    NUM_METRICS,
} MediumPerformanceMetric;

typedef struct DifferenceMetrics {
    double mean_abs_err;
    double mean_rel_err;
} DiffMetrics;

typedef struct performance_metrics {
    double time;
    double flops;
    double speedup;
    double efficiency;
} PerformanceMetrics;

struct DifferenceMetrics computeDifferenceMetrics(
    const double* ref,
    const double* res,
    int n,
    double abs_tol,
    double rel_tol,
    bool print_summary
);

void initialize_metrics();
void cleanup_metrics();
double get_metric_value(MediumPerformanceMetric type);
double get_metric_min(MediumPerformanceMetric type);
double get_metric_max(MediumPerformanceMetric type);
double get_metric_error(MediumPerformanceMetric type);
double get_metric_stddev(MediumPerformanceMetric type);
double get_metric_variance(MediumPerformanceMetric type);
void update_medium_metric(MediumPerformanceMetric type, double value);
void update_metric_error(const MediumPerformanceMetric type, const double error_value);
void reset_medium_time_metrics();
DiffMetrics computeAverageErrors(const MediumPerformanceMetric type);
void accumulateErrors(const DiffMetrics* iteration_metrics, const MediumPerformanceMetric type);
double calculate_flops(int nz, double time);
void print_flops(double flops);
#endif //PERFORMANCE_CALCULATE_H
