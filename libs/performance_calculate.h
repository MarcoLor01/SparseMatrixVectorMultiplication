//
// Created by HP on 09/03/2025.
//

#ifndef PERFORMANCE_CALCULATE_H
#define PERFORMANCE_CALCULATE_H
#include <stddef.h>

typedef enum {
    SERIAL_TIME,
    PARALLEL_CSR_TIME,
    PARALLEL_SIMD_CSR_TIME,
    PARALLEL_HLL_TIME,
    PARALLEL_HLL_SIMD_TIME,
    SERIAL_HLL_TIME,
    NUM_METRICS,
} MediumPerformanceMetric;

double get_metric_value(MediumPerformanceMetric type);
void update_medium_metric(MediumPerformanceMetric type, double value);
double calculate_flops(int nz, double time);
void reset_medium_time_metrics();
void print_flops(double flops);
#endif //PERFORMANCE_CALCULATE_H
