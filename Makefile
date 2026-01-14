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

# По умолчанию компиляция без CUDA
all: $(TARGET)

# Компиляция с CUDA (если доступна)
cuda: CXXFLAGS += -DUSE_CUDA
cuda: $(TARGET)_cuda
	@echo "CUDA версия скомпилирована: $(TARGET)_cuda"
	@ln -sf $(TARGET)_cuda $(TARGET)

# Компиляция без CUDA
$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)
	@echo "CPU версия скомпилирована: $(TARGET)"

# Компиляция с CUDA
$(TARGET)_cuda: $(SOURCE) $(CUDA_OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET)_cuda $(SOURCE) $(CUDA_OBJ) -L/usr/local/cuda/lib64 -lcudart -lcuda

# Компиляция CUDA модуля
$(CUDA_OBJ): $(CUDA_SOURCE)
	$(NVCC) -c $(CUDA_SOURCE) -o $(CUDA_OBJ) --compiler-options -fPIC

clean:
	rm -f $(TARGET) $(TARGET)_cuda *.o steam_analysis.out steam_analysis.err analysis_results.txt analysis_results.csv

install:
	@if [ -f $(TARGET)_cuda ]; then \
		cp $(TARGET)_cuda /mnt/share/$(TARGET); \
		echo "CUDA версия установлена в /mnt/share/"; \
	else \
		cp $(TARGET) /mnt/share/; \
		echo "CPU версия установлена в /mnt/share/"; \
	fi

.PHONY: all cuda clean install

