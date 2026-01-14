#ifndef CUDA_ACCELERATOR_H
#define CUDA_ACCELERATOR_H

// Вычисление отношений отзывов с использованием CUDA
// positive, negative - входные массивы
// ratios - выходной массив
// count - количество элементов
bool cuda_calculate_review_ratios(int* positive, int* negative, double* ratios, int count);

// Вычисление среднего значения с использованием CUDA
// values - входной массив
// count - количество элементов
// возвращает среднее значение
double cuda_calculate_average(double* values, int count);

// Сортировка издателей с использованием Thrust
// dlc_ratios - отношения DLC/игры
// indices - индексы после сортировки
// count - количество элементов
bool cuda_sort_publishers(double* dlc_ratios, int* indices, int count);

// Проверка доступности CUDA
bool cuda_is_available();

// Инициализация CUDA устройства
bool cuda_initialize();

// Очистка CUDA ресурсов
void cuda_cleanup();

#endif // CUDA_ACCELERATOR_H
