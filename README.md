# ==============================
# CUDA N-Body GPU Makefile AI README
# ==============================

# Compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -O3 -arch=sm_61

# Target executable
TARGET = nbody

# Source file
SRC = nbody.cu

# Default target
all: $(TARGET)

# Build CUDA executable
$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

# Run a small test
run: $(TARGET)
	./$(TARGET) 1000 0.01 5 1000 128

# Run required GPU benchmark cases
bench: $(TARGET)
	mkdir -p results

	/usr/bin/time -f "1000 GPU Runtime: %e sec" \
	./$(TARGET) 1000 0.01 10 1000 128 \
	> results/gpu_1000.log \
	2> results/gpu_1000.time

	/usr/bin/time -f "10000 GPU Runtime: %e sec" \
	./$(TARGET) 10000 0.01 10 10000 128 \
	> results/gpu_10000.log \
	2> results/gpu_10000.time

	/usr/bin/time -f "100000 GPU Runtime: %e sec" \
	./$(TARGET) 100000 0.01 10 100000 128 \
	> results/gpu_100000.log \
	2> results/gpu_100000.time

# Remove generated files
clean:
	rm -f $(TARGET)
	rm -rf results

.PHONY: all run bench clean