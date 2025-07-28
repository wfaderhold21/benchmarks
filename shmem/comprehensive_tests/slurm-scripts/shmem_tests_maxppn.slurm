#!/bin/bash
#SBATCH --job-name=shmem_tests_maxppn
#SBATCH --output=shmem_tests_maxppn_%j.out
#SBATCH --error=shmem_tests_maxppn_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=compute
##SBATCH --account=your_account
##SBATCH --qos=normal

# SHMEM Comprehensive Tests - Maximum Processes Per Node (MAX PPN)
# This configuration tests intra-node and mixed communication performance
# Modify the SBATCH directives above as needed for your system

echo "=================================================="
echo "SHMEM Comprehensive Tests - MAX PPN Configuration"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "Total Tasks: $SLURM_NTASKS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Start Time: $(date)"
echo "=================================================="

# Print node information
echo "Allocated Nodes:"
scontrol show hostnames $SLURM_JOB_NODELIST
echo ""

# Auto-detect maximum PPN if not explicitly set
if [ -z "$SLURM_NTASKS_PER_NODE" ] || [ "$SLURM_NTASKS_PER_NODE" = "1" ]; then
    # Try to determine max PPN from available CPUs
    CPUS_PER_NODE=$(scontrol show node $(scontrol show hostnames $SLURM_JOB_NODELIST | head -1) | grep CPUTot | awk -F= '{print $2}' | awk '{print $1}')
    if [ -n "$CPUS_PER_NODE" ]; then
        MAX_PPN=$CPUS_PER_NODE
        echo "Auto-detected MAX PPN: $MAX_PPN (based on CPUs per node)"
    else
        MAX_PPN=16  # Default fallback
        echo "Using default MAX PPN: $MAX_PPN"
    fi
else
    MAX_PPN=$SLURM_NTASKS_PER_NODE
    echo "Using configured PPN: $MAX_PPN"
fi

echo "Configuration: $MAX_PPN processes per node on $SLURM_JOB_NUM_NODES nodes"
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

# Test parameters for MAX PPN configuration
# Use smaller message sizes since we have more processes
MAX_SIZE=262144  # 256KB - smaller for higher process counts
ITERATIONS=5000  # Fewer iterations due to more processes
PE_COUNT=$SLURM_NTASKS

echo "Test Configuration:"
echo "- PE Count: $PE_COUNT"
echo "- Max Size: $MAX_SIZE bytes"
echo "- Iterations: $ITERATIONS"
echo "- Configuration: MAX PPN (intra-node + inter-node)"
echo "- Expected communication patterns:"
echo "  * Intra-node: Fast shared memory"
echo "  * Inter-node: Network communication"
echo ""

# Create results directory
RESULTS_DIR="results_maxppn_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

# Save job information
cat > $RESULTS_DIR/job_info.txt << EOF
SLURM Job Information - MAX PPN Configuration
============================================
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

Configuration: Maximum Processes Per Node (MAX PPN)
Purpose: Mixed intra-node and inter-node communication testing
Max Message Size: $MAX_SIZE bytes
Iterations: $ITERATIONS

Communication Patterns:
- Intra-node: Shared memory (faster)
- Inter-node: Network (slower)
- Mixed patterns in collective operations
EOF

# Function to run a test and capture output
run_test() {
    local test_name="$1"
    local executable="$2"
    local args="$3"
    local output_file="$RESULTS_DIR/${test_name}_maxppn.out"
    local error_file="$RESULTS_DIR/${test_name}_maxppn.err"
    
    echo "Running $test_name with $PE_COUNT PEs (MAX PPN)..."
    echo "Command: mpirun -np $PE_COUNT $executable $args"
    
    # Use process binding for better performance with MAX PPN
    local mpi_opts=""
    if command -v mpirun >/dev/null 2>&1; then
        # Add binding options if supported
        mpi_opts="--bind-to core --map-by node:PE=1"
    fi
    
    if timeout 600 mpirun $mpi_opts -np $PE_COUNT $executable $args > $output_file 2> $error_file; then
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
echo "Running Point-to-Point Tests (MAX PPN)"
echo "=========================================="

run_test "put_benchmark" "./point_to_point/shmem_put_bench" "--max-size $MAX_SIZE --iterations $ITERATIONS"
run_test "get_benchmark" "./point_to_point/shmem_get_bench" "--max-size $MAX_SIZE --iterations $ITERATIONS"

# Run Atomic Operations Tests
echo "=========================================="
echo "Running Atomic Operations Tests (MAX PPN)"
echo "=========================================="

# Reduce iterations for atomic tests with many processes
ATOMIC_ITERATIONS=$((ITERATIONS / 2))
run_test "atomic_benchmark" "./atomic/shmem_atomic_bench" "--iterations $ATOMIC_ITERATIONS"

# Run Collective Operations Tests  
echo "=========================================="
echo "Running Collective Operations Tests (MAX PPN)"
echo "=========================================="

# Use smaller sizes and fewer iterations for collectives with many processes
COLLECTIVE_SIZE=$((MAX_SIZE / 8))
COLLECTIVE_ITERATIONS=$((ITERATIONS / 20))

run_test "collective_benchmark" "./collective/shmem_collective_bench" "--max-size $COLLECTIVE_SIZE --iterations $COLLECTIVE_ITERATIONS"

# Additional test: Communication pattern analysis
echo "=========================================="
echo "Communication Pattern Analysis"
echo "=========================================="

echo "Analyzing intra-node vs inter-node communication patterns..."

# Create a simple analysis of the results
ANALYSIS_FILE="$RESULTS_DIR/communication_analysis.txt"
cat > $ANALYSIS_FILE << EOF
Communication Pattern Analysis - MAX PPN Configuration
=====================================================
Job ID: $SLURM_JOB_ID
Analysis Time: $(date)

Configuration Details:
- Nodes: $SLURM_JOB_NUM_NODES
- Processes per Node: $SLURM_NTASKS_PER_NODE
- Total Processes: $SLURM_NTASKS

Expected Communication Breakdown:
- Intra-node pairs: $((SLURM_NTASKS_PER_NODE * (SLURM_NTASKS_PER_NODE - 1) * SLURM_JOB_NUM_NODES / 2))
- Inter-node pairs: $((SLURM_NTASKS_PER_NODE * SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES * (SLURM_JOB_NUM_NODES - 1) / 2))

Performance Considerations:
- Intra-node communication should show higher bandwidth
- Inter-node communication should show higher latency
- Collective operations will use mixed patterns
- Atomic operations may show contention effects

EOF

echo "Communication analysis saved to: $ANALYSIS_FILE"

# Generate summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="

SUMMARY_FILE="$RESULTS_DIR/summary_maxppn.txt"
cat > $SUMMARY_FILE << EOF
SHMEM Comprehensive Tests Summary - MAX PPN Configuration
========================================================
Job ID: $SLURM_JOB_ID
Completion Time: $(date)
Total Runtime: $SECONDS seconds

Configuration:
- Nodes: $SLURM_JOB_NUM_NODES
- Tasks per Node: $SLURM_NTASKS_PER_NODE  
- Total PEs: $SLURM_NTASKS
- Focus: Mixed intra-node and inter-node communication

Test Results:
EOF

# Check test results
TOTAL_TESTS=0
PASSED_TESTS=0

for result_file in $RESULTS_DIR/*_maxppn.out; do
    if [ -f "$result_file" ]; then
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        test_name=$(basename "$result_file" _maxppn.out)
        
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

Performance Notes:
- Intra-node communication benefits from shared memory
- Inter-node communication uses network fabric
- Collective operations show mixed performance patterns
- Higher process counts may show different scaling behavior

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