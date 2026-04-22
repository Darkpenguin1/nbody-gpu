#!/bin/bash
#SBATCH --job-name=nbody_bench
#SBATCH --partition=GPU
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=bench_%j.out
#SBATCH --error=bench_%j.err

module purge
module load cuda/12.4

# Recompile fresh each run
rm -f nbody
nvcc -O3 -arch=sm_61 -o nbody nbody.cu

# Create results directory if it does not exist
mkdir -p results

# Timestamped summary file
SUMMARY_FILE="results/benchmark_summary_$(date +%Y%m%d_%H%M%S).txt"

echo "===== GPU N-Body Benchmark Results =====" | tee -a "$SUMMARY_FILE"
echo "Node: $(hostname)" | tee -a "$SUMMARY_FILE"
echo "Date: $(date)" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

run_test () {
    PARTICLES=$1

    LOG_FILE="results/gpu_${PARTICLES}.log"

    echo "Running GPU test with ${PARTICLES} particles..." | tee -a "$SUMMARY_FILE"

    /usr/bin/time -f "Runtime: %e seconds" \
        ./nbody ${PARTICLES} 0.01 10 ${PARTICLES} 128 \
        > "$LOG_FILE" 2>> "$SUMMARY_FILE"

    echo "Output written to: $LOG_FILE" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"
}

# Required benchmark cases
run_test 1000
run_test 10000
run_test 100000

echo "All GPU benchmark tests completed." | tee -a "$SUMMARY_FILE"

