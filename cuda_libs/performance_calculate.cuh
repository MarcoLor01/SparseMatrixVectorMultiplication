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

double get_metric_value(MediumPerformanceMetric type);
void update_medium_metric(MediumPerformanceMetric type, double value);
double calculate_flops(int nz, double time);
void reset_medium_time_metrics();
void print_flops(double flops);
#endif //PERFORMANCE_CALCULATE_H
