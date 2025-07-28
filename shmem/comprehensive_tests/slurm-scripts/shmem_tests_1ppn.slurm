#!/bin/bash
#SBATCH --job-name=shmem_tests_1ppn
#SBATCH --output=shmem_tests_1ppn_%j.out
#SBATCH --error=shmem_tests_1ppn_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=compute
##SBATCH --account=your_account
##SBATCH --qos=normal

# SHMEM Comprehensive Tests - 1 Process Per Node (1 PPN)
# This configuration tests inter-node communication performance
# Modify the SBATCH directives above as needed for your system

echo "=================================================="
echo "SHMEM Comprehensive Tests - 1 PPN Configuration"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "Total Tasks: $SLURM_NTASKS"
echo "Start Time: $(date)"
echo "=================================================="

# Print node information
echo "Allocated Nodes:"
scontrol show hostnames $SLURM_JOB_NODELIST
echo ""

# Load necessary modules (adjust for your system)
# module load openmpi/4.1.0
# module load shmem

# Set up environment - adjust paths as needed
export PATH=/home/ferrol/install.ompi/bin:$PATH
export LD_LIBRARY_PATH=/home/ferrol/install.ompi/lib:$LD_LIBRARY_PATH

# Change to the test directory
cd $SLURM_SUBMIT_DIR

echo "Current directory: $(pwd)"
echo "Available executables:"
find . -name "*bench*" -executable | sort
echo ""

# Verify SHMEM environment
echo "SHMEM Environment Check:"
echo "shmemcc: $(which shmemcc)"
echo "mpirun: $(which mpirun)"
echo ""

# Build tests if needed
echo "Building tests..."
if ! make all; then
    echo "ERROR: Failed to build tests"
    exit 1
fi
echo "Build completed successfully"
echo ""

# Test parameters for 1 PPN configuration
MAX_SIZE=1048576  # 1MB - larger sizes for inter-node testing
ITERATIONS=10000
PE_COUNT=$SLURM_NTASKS

echo "Test Configuration:"
echo "- PE Count: $PE_COUNT"
echo "- Max Size: $MAX_SIZE bytes"
echo "- Iterations: $ITERATIONS"
echo "- Configuration: 1 PPN (inter-node focus)"
echo ""

# Create results directory
RESULTS_DIR="results_1ppn_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

# Save job information
cat > $RESULTS_DIR/job_info.txt << EOF
SLURM Job Information - 1 PPN Configuration
===========================================
Job ID: $SLURM_JOB_ID
Job Name: $SLURM_JOB_NAME
Nodes: $SLURM_JOB_NUM_NODES
Tasks per Node: $SLURM_NTASKS_PER_NODE
Total Tasks: $SLURM_NTASKS
CPUs per Task: $SLURM_CPUS_PER_TASK
Partition: $SLURM_JOB_PARTITION
Start Time: $(date)
Submit Directory: $SLURM_SUBMIT_DIR
Node List: $SLURM_JOB_NODELIST

Configuration: 1 Process Per Node (1 PPN)
Purpose: Inter-node communication testing
Max Message Size: $MAX_SIZE bytes
Iterations: $ITERATIONS
EOF

# Function to run a test and capture output
run_test() {
    local test_name="$1"
    local executable="$2"
    local args="$3"
    local output_file="$RESULTS_DIR/${test_name}_1ppn.out"
    local error_file="$RESULTS_DIR/${test_name}_1ppn.err"
    
    echo "Running $test_name with $PE_COUNT PEs (1 PPN)..."
    echo "Command: mpirun -np $PE_COUNT $executable $args"
    
    if timeout 600 mpirun -np $PE_COUNT $executable $args > $output_file 2> $error_file; then
        echo "✓ $test_name completed successfully"
        # Show brief results
        echo "  Results preview:"
        tail -5 $output_file | sed 's/^/    /'
    else
        echo "✗ $test_name failed"
        if [ -s $error_file ]; then
            echo "  Error output:"
            tail -5 $error_file | sed 's/^/    /'
        fi
    fi
    echo ""
}

# Run Point-to-Point Tests
echo "=========================================="
echo "Running Point-to-Point Tests (1 PPN)"
echo "=========================================="

run_test "put_benchmark" "./point_to_point/shmem_put_bench" "--max-size $MAX_SIZE --iterations $ITERATIONS"
run_test "get_benchmark" "./point_to_point/shmem_get_bench" "--max-size $MAX_SIZE --iterations $ITERATIONS"

# Run Atomic Operations Tests
echo "=========================================="
echo "Running Atomic Operations Tests (1 PPN)"
echo "=========================================="

run_test "atomic_benchmark" "./atomic/shmem_atomic_bench" "--iterations $ITERATIONS"

# Run Collective Operations Tests  
echo "=========================================="
echo "Running Collective Operations Tests (1 PPN)"
echo "=========================================="

# Use smaller sizes and fewer iterations for collectives
COLLECTIVE_SIZE=$((MAX_SIZE / 4))
COLLECTIVE_ITERATIONS=$((ITERATIONS / 10))

run_test "collective_benchmark" "./collective/shmem_collective_bench" "--max-size $COLLECTIVE_SIZE --iterations $COLLECTIVE_ITERATIONS"

# Generate summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="

SUMMARY_FILE="$RESULTS_DIR/summary_1ppn.txt"
cat > $SUMMARY_FILE << EOF
SHMEM Comprehensive Tests Summary - 1 PPN Configuration
======================================================
Job ID: $SLURM_JOB_ID
Completion Time: $(date)
Total Runtime: $SECONDS seconds

Configuration:
- Nodes: $SLURM_JOB_NUM_NODES
- Tasks per Node: $SLURM_NTASKS_PER_NODE  
- Total PEs: $SLURM_NTASKS
- Focus: Inter-node communication (1 PPN)

Test Results:
EOF

# Check test results
TOTAL_TESTS=0
PASSED_TESTS=0

for result_file in $RESULTS_DIR/*_1ppn.out; do
    if [ -f "$result_file" ]; then
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        test_name=$(basename "$result_file" _1ppn.out)
        
        if [ -s "$result_file" ] && ! grep -q "ERROR\|FAIL" "$result_file"; then
            PASSED_TESTS=$((PASSED_TESTS + 1))
            echo "✓ $test_name: PASSED" >> $SUMMARY_FILE
        else
            echo "✗ $test_name: FAILED" >> $SUMMARY_FILE
        fi
    fi
done

cat >> $SUMMARY_FILE << EOF

Summary:
- Total Tests: $TOTAL_TESTS
- Passed: $PASSED_TESTS
- Failed: $((TOTAL_TESTS - PASSED_TESTS))
- Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%

Results Directory: $RESULTS_DIR
EOF

echo ""
cat $SUMMARY_FILE

# Copy results to a standard location if desired
# cp -r $RESULTS_DIR /shared/results/ 2>/dev/null || true

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "Total runtime: $SECONDS seconds"
echo "Results saved in: $RESULTS_DIR"
echo "==========================================" 