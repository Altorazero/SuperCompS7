#include "cuda_accelerator.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <cmath>

// CUDA kernel для вычисления отношений отзывов
__global__ void calculateReviewRatiosKernel(int* positive, int* negative, 
                                            double* ratios, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (negative[idx] > 0) {
            ratios[idx] = static_cast<double>(positive[idx]) / static_cast<double>(negative[idx]);
        } else if (positive[idx] > 0) {
            ratios[idx] = static_cast<double>(positive[idx]);
        } else {
            ratios[idx] = 0.0;
        }
    }
}

// CUDA kernel для суммирования массива (reduction)
__global__ void sumReductionKernel(double* input, double* output, int n) {
    extern __shared__ double sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0.0;
    __syncthreads();
    
    // Reduction в shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Вычисление отношений отзывов с использованием CUDA
bool cuda_calculate_review_ratios(int* positive, int* negative, double* ratios, int count) {
    if (count <= 0) return false;
    
    // Выделяем память на GPU
    int* d_positive;
    int* d_negative;
    double* d_ratios;
    
    cudaError_t err;
    
    err = cudaMalloc(&d_positive, count * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error (positive): " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaMalloc(&d_negative, count * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_positive);
        std::cerr << "CUDA malloc error (negative): " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaMalloc(&d_ratios, count * sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(d_positive);
        cudaFree(d_negative);
        std::cerr << "CUDA malloc error (ratios): " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Копируем данные на GPU
    cudaMemcpy(d_positive, positive, count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_negative, negative, count * sizeof(int), cudaMemcpyHostToDevice);
    
    // Запускаем kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    
    calculateReviewRatiosKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_positive, d_negative, d_ratios, count
    );
    
    // Проверяем ошибки запуска kernel
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_positive);
        cudaFree(d_negative);
        cudaFree(d_ratios);
        return false;
    }
    
    // Копируем результаты обратно
    cudaMemcpy(ratios, d_ratios, count * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Освобождаем память
    cudaFree(d_positive);
    cudaFree(d_negative);
    cudaFree(d_ratios);
    
    return true;
}

// Вычисление среднего значения с использованием CUDA
double cuda_calculate_average(double* values, int count) {
    if (count <= 0) return 0.0;
    
    // Выделяем память на GPU
    double* d_input;
    double* d_output;
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaMalloc(&d_input, count * sizeof(double));
    cudaMalloc(&d_output, blocksPerGrid * sizeof(double));
    
    // Копируем данные на GPU
    cudaMemcpy(d_input, values, count * sizeof(double), cudaMemcpyHostToDevice);
    
    // Запускаем kernel для суммирования
    size_t sharedMemSize = threadsPerBlock * sizeof(double);
    sumReductionKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        d_input, d_output, count
    );
    
    // Копируем частичные суммы обратно
    double* partial_sums = new double[blocksPerGrid];
    cudaMemcpy(partial_sums, d_output, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Суммируем частичные суммы на CPU
    double total_sum = 0.0;
    for (int i = 0; i < blocksPerGrid; i++) {
        total_sum += partial_sums[i];
    }
    
    delete[] partial_sums;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return total_sum / count;
}

// Структура для сортировки издателей
struct PublisherForSort {
    double dlc_ratio;
    int index;
    
    __host__ __device__
    bool operator<(const PublisherForSort& other) const {
        return dlc_ratio > other.dlc_ratio; // Сортировка по убыванию
    }
};

// Сортировка издателей с использованием Thrust
bool cuda_sort_publishers(double* dlc_ratios, int* indices, int count) {
    if (count <= 0) return false;
    
    try {
        // Создаем векторы на GPU
        thrust::device_vector<PublisherForSort> d_publishers(count);
        
        // Копируем данные
        std::vector<PublisherForSort> h_publishers(count);
        for (int i = 0; i < count; i++) {
            h_publishers[i].dlc_ratio = dlc_ratios[i];
            h_publishers[i].index = i;
        }
        
        d_publishers = h_publishers;
        
        // Сортируем на GPU
        thrust::sort(d_publishers.begin(), d_publishers.end());
        
        // Копируем результаты обратно
        thrust::copy(d_publishers.begin(), d_publishers.end(), h_publishers.begin());
        
        for (int i = 0; i < count; i++) {
            indices[i] = h_publishers[i].index;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "CUDA sort error: " << e.what() << std::endl;
        return false;
    }
}

// Проверка доступности CUDA
bool cuda_is_available() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess || deviceCount == 0) {
        return false;
    }
    
    return true;
}

// Инициализация CUDA устройства
bool cuda_initialize() {
    if (!cuda_is_available()) {
        return false;
    }
    
    // Устанавливаем устройство 0
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "CUDA set device error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

// Очистка CUDA ресурсов
void cuda_cleanup() {
    cudaDeviceReset();
}
