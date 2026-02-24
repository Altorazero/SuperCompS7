# Makefile для компиляции MPI программы анализа игр Steam

CXX = mpic++
NVCC = nvcc
CXXFLAGS = -std=c++11 -O2 -Wall
TARGET = steam_analysis
SOURCE = steam_analysis.cpp
CUDA_SOURCE = cuda_accelerator.cu
CUDA_OBJ = cuda_accelerator.o

# Определяем наличие CUDA
CUDA_AVAILABLE := $(shell which nvcc 2>/dev/null)

# По умолчанию компиляция без CUDA (для совместимости со всеми узлами)
all: $(TARGET)

# Компиляция с CUDA - создает отдельную библиотеку
cuda: $(TARGET)_with_cuda
	@echo "CUDA версия скомпилирована: $(TARGET)_with_cuda"
	@echo "Используйте 'make install-cuda' для установки CUDA версии"
	@echo "Или 'make install' для установки CPU версии (работает на всех узлах)"

# Компиляция без CUDA - работает на всех узлах
$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)
	@echo "CPU версия скомпилирована: $(TARGET) (работает на всех узлах)"

# Компиляция с CUDA - только для GPU узлов
$(TARGET)_with_cuda: CXXFLAGS += -DUSE_CUDA
$(TARGET)_with_cuda: $(SOURCE) $(CUDA_OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET)_with_cuda $(SOURCE) $(CUDA_OBJ) -L/usr/local/cuda/lib64 -lcudart -lcuda
	@echo "CUDA версия требует CUDA библиотеки на всех узлах!"

# Компиляция CUDA модуля
$(CUDA_OBJ): $(CUDA_SOURCE)
	$(NVCC) -c $(CUDA_SOURCE) -o $(CUDA_OBJ) --compiler-options -fPIC

clean:
	rm -f $(TARGET) $(TARGET)_with_cuda *.o steam_analysis.out steam_analysis.err analysis_results.txt analysis_results.csv

# Установка CPU версии (рекомендуется - работает на всех узлах)
install:
	cp $(TARGET) /mnt/share/
	@echo "CPU версия установлена в /mnt/share/"
	@echo "Эта версия работает на всех узлах (GPU и CPU)"

# Установка CUDA версии (только если CUDA доступна на ВСЕХ узлах включая головной)
install-cuda:
	@if [ -f $(TARGET)_with_cuda ]; then \
		cp $(TARGET)_with_cuda /mnt/share/$(TARGET); \
		echo "ВНИМАНИЕ: CUDA версия установлена в /mnt/share/"; \
		echo "Требуется CUDA библиотеки на ВСЕХ узлах!"; \
	else \
		echo "Ошибка: Сначала выполните 'make cuda'"; \
		exit 1; \
	fi

.PHONY: all cuda clean install install-cuda

