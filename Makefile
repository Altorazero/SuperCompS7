# Makefile для компиляции MPI программы анализа игр Steam

CXX = mpic++
CXXFLAGS = -std=c++11 -O2 -Wall
TARGET = steam_analysis
SOURCE = steam_analysis.cpp

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET) *.o steam_analysis.out steam_analysis.err analysis_results.txt analysis_results.csv

install:
	cp $(TARGET) /mnt/share/

.PHONY: all clean install
