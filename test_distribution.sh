#!/bin/bash
# Test script to demonstrate weighted workload distribution

echo "=== Демонстрация взвешенного распределения данных ==="
echo ""
echo "Конфигурация кластера:"
echo "  - 4 GPU процесса (вес = 3 каждый)"
echo "  - 4 CPU процесса (вес = 1 каждый)"
echo "  - Общий вес = 4*3 + 4*1 = 16"
echo ""
echo "Распределение для первых 32 строк (2 полных цикла):"
echo "Строка | Позиция в цикле | Кумулятивный вес | Процесс | Тип"
echo "-------|-----------------|------------------|---------|-----"

# Симуляция алгоритма распределения
TOTAL_WEIGHT=16
WEIGHTS=(3 3 3 3 1 1 1 1)  # 4 GPU + 4 CPU процесса

for LINE in {0..31}; do
    WEIGHT_POS=$((LINE % TOTAL_WEIGHT))
    CUMULATIVE=0
    ASSIGNED_PROCESS=""
    PROCESS_TYPE=""
    
    for PROC in {0..7}; do
        CUMULATIVE=$((CUMULATIVE + WEIGHTS[PROC]))
        if [ $WEIGHT_POS -lt $CUMULATIVE ] && [ -z "$ASSIGNED_PROCESS" ]; then
            ASSIGNED_PROCESS=$PROC
            if [ $PROC -lt 4 ]; then
                PROCESS_TYPE="GPU"
            else
                PROCESS_TYPE="CPU"
            fi
            break
        fi
    done
    
    printf "%6d | %15d | %16d | %7d | %s\n" $LINE $WEIGHT_POS $CUMULATIVE $ASSIGNED_PROCESS $PROCESS_TYPE
done

echo ""
echo "=== Статистика распределения для 1000 строк ==="

# Подсчет для 1000 строк
declare -A COUNTS
for PROC in {0..7}; do
    COUNTS[$PROC]=0
done

for LINE in {0..999}; do
    WEIGHT_POS=$((LINE % TOTAL_WEIGHT))
    CUMULATIVE=0
    
    for PROC in {0..7}; do
        CUMULATIVE=$((CUMULATIVE + WEIGHTS[PROC]))
        if [ $WEIGHT_POS -lt $CUMULATIVE ]; then
            COUNTS[$PROC]=$((COUNTS[$PROC] + 1))
            break
        fi
    done
done

echo ""
echo "Процесс | Тип | Вес | Строк | Процент"
echo "--------|-----|-----|-------|--------"

GPU_TOTAL=0
CPU_TOTAL=0

for PROC in {0..7}; do
    COUNT=${COUNTS[$PROC]}
    WEIGHT=${WEIGHTS[$PROC]}
    PERCENT=$(echo "scale=2; $COUNT * 100 / 1000" | bc)
    
    if [ $PROC -lt 4 ]; then
        TYPE="GPU"
        GPU_TOTAL=$((GPU_TOTAL + COUNT))
    else
        TYPE="CPU"
        CPU_TOTAL=$((CPU_TOTAL + COUNT))
    fi
    
    printf "%7d | %3s | %3d | %5d | %6.2f%%\n" $PROC "$TYPE" $WEIGHT $COUNT $PERCENT
done

echo "--------|-----|-----|-------|--------"
GPU_PERCENT=$(echo "scale=2; $GPU_TOTAL * 100 / 1000" | bc)
CPU_PERCENT=$(echo "scale=2; $CPU_TOTAL * 100 / 1000" | bc)
printf "GPU итого:       | %5d | %6.2f%%\n" $GPU_TOTAL $GPU_PERCENT
printf "CPU итого:       | %5d | %6.2f%%\n" $CPU_TOTAL $CPU_PERCENT

echo ""
echo "=== Результат ==="
echo "✓ GPU узлы обрабатывают $GPU_TOTAL строк (${GPU_PERCENT}%)"
echo "✓ CPU узлы обрабатывают $CPU_TOTAL строк (${CPU_PERCENT}%)"
echo "✓ Соотношение GPU:CPU = $(echo "scale=2; $GPU_TOTAL / $CPU_TOTAL" | bc):1"
echo ""
