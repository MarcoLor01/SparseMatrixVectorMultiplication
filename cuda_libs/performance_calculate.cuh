//
// Created by HP on 09/03/2025.
//

#ifndef PERFORMANCE_CALCULATE_CUH
#define PERFORMANCE_CALCULATE_CUH
#include <stddef.h>

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
DifferenceMetrics computeDifferenceMetrics(
    const double* ref,
    const double* res,
    int n,
    double abs_tol = 1e-5,
    double rel_tol = 1e-4,
    bool print_summary = true
);
#endif //PERFORMANCE_CALCULATE_H
