//
// Created by HP on 09/03/2025.
//

#ifndef PERFORMANCE_CALCULATE_H
#define PERFORMANCE_CALCULATE_H
#include <stddef.h>

typedef enum {
    SERIAL_TIME,
    PARALLEL_CSR_TIME,
    PARALLEL_UNROLLING_CSR_TIME,
    PARALLEL_ELLPACK_TIME,
    PARALLEL_HLL_SIMD_TIME,
    SERIAL_HLL_TIME,
    NUM_METRICS,
} MediumPerformanceMetric;

double get_metric_value(MediumPerformanceMetric type);
void update_medium_metric(MediumPerformanceMetric type, double value);
double calculate_flops(int nz, double time);
void calculate_and_print_thread_averages(int num_thread, const char *output_summary_file);
void reset_medium_time_metrics();
void print_flops(double flops);
#endif //PERFORMANCE_CALCULATE_H
