//
// Created by HP on 09/03/2025.
//

#include <stdio.h>
#include "performance_calculate.h"
#include <stdlib.h>
#include <math.h>
#include <utility.h>

MetricStats metrics[NUM_METRICS];

void initialize_metrics() {
    for (int i = 0; i < NUM_METRICS; i++) {
        metrics[i].sum = 0.0;
        metrics[i].min = INFINITY;
        metrics[i].max = -INFINITY;
        metrics[i].count = 0;
        metrics[i].capacity = INITIAL_CAPACITY;
        metrics[i].error = 0.0;
        metrics[i].values = (double*)malloc(INITIAL_CAPACITY * sizeof(double));
        if (!metrics[i].values) {
            fprintf(stderr, "Failed to allocate memory for metrics\n");
            exit(EXIT_FAILURE);
        }
    }
}

void cleanup_metrics() {
    for (int i = 0; i < NUM_METRICS; i++) {
        FREE_CHECK(metrics[i].values);
        metrics[i].values = NULL;
    }
}

// Funzione per ottenere il valore medio di una metrica
double get_metric_value(const MediumPerformanceMetric type) {
    if (metrics[type].count == 0) return 0.0;
    return metrics[type].sum / metrics[type].count;
}

// Funzione per ottenere il valore minimo
double get_metric_min(const MediumPerformanceMetric type) {
    if (metrics[type].count == 0) return 0.0;
    return metrics[type].min;
}

// Funzione per ottenere il valore massimo
double get_metric_max(const MediumPerformanceMetric type) {
    if (metrics[type].count == 0) return 0.0;
    return metrics[type].max;
}

// Funzione per calcolare la varianza
double get_metric_variance(const MediumPerformanceMetric type) {
    if (metrics[type].count <= 1) return 0.0;

    double mean = metrics[type].sum / metrics[type].count;
    double sum_sq_diff = 0.0;

    for (int i = 0; i < metrics[type].count; i++) {
        double diff = metrics[type].values[i] - mean;
        sum_sq_diff += diff * diff;
    }

    return sum_sq_diff / metrics[type].count;
}

// Funzione per calcolare la deviazione standard
double get_metric_stddev(const MediumPerformanceMetric type) {
    return sqrt(get_metric_variance(type));
}

// Funzione per il calcolo dell'errore medio
double get_metric_error(const MediumPerformanceMetric type) {
    if (metrics[type].count == 0) return 0.0;
    return metrics[type].error;
}

// Funzione per aggiornare l'errore
void update_metric_error(const MediumPerformanceMetric type, const double error_value) {
    int real_error_count = metrics[type].count + 1;

    if (real_error_count == 1) {
        // Prima misurazione dell'errore
        metrics[type].error = error_value;
    } else {
        // Media cumulativa utilizzando il conteggio corretto
        metrics[type].error = ((metrics[type].error * (real_error_count - 1)) + error_value)
                             / real_error_count;
    }
}

// Funzione per aggiornare una metrica
void update_medium_metric(const MediumPerformanceMetric type, const double value) {
    metrics[type].sum += value;

    if (value < metrics[type].min) {
        printf("Type: %d, Value: %.10f\n", type, value);
        metrics[type].min = value;
    }

    if (value > metrics[type].max) {
        metrics[type].max = value;
    }

    if (metrics[type].count >= metrics[type].capacity) {
        metrics[type].capacity *= 2;
        metrics[type].values = (double*)realloc(metrics[type].values,
                                              metrics[type].capacity * sizeof(double));
        if (!metrics[type].values) {
            fprintf(stderr, "Failed to reallocate memory for metrics\n");
            exit(EXIT_FAILURE);
        }
    }

    // Aggiungi il nuovo valore all'array
    metrics[type].values[metrics[type].count] = value;
    metrics[type].count++;
}

void reset_medium_time_metrics() {
    for (int i = 0; i < NUM_METRICS; i++) {
        metrics[i].sum = 0.0;
        metrics[i].min = INFINITY;
        metrics[i].max = -INFINITY;
        metrics[i].count = 0;
        metrics[i].error = 0.0;
    }
}

// Funzione per calcolare i FLOPS
double calculate_flops(const int nz, const double time) {
    // 2 * NZ / tempo (in secondi) => risultato in FLOPS
    return 2.0 * nz / time;
}

// Funzione per convertire e stampare i FLOPS in formato leggibile
void print_flops(double flops) {
    const char *units[] = {"FLOPS", "KFLOPS", "MFLOPS", "GFLOPS", "TFLOPS", "PFLOPS", "EFLOPS"};
    int unit_index = 0;

    while (flops >= 1000.0 && unit_index < 6) {
        flops /= 1000.0;
        unit_index++;
    }

    printf("%.3f %s\n", flops, units[unit_index]);
}

