#!/bin/bash
#SBATCH --job-name=shmem_tests_auto
#SBATCH --output=shmem_tests_auto_%j.out
#SBATCH --error=shmem_tests_auto_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=compute
##SBATCH --account=your_account
##SBATCH --qos=normal

# SHMEM Comprehensive Tests - Adaptive Configuration
# This script automatically selects appropriate test parameters based on the SLURM configuration
# and can run both 1 PPN and MAX PPN tests in sequence for comparison
# 
# Usage:
#   sbatch shmem_tests_auto.slurm                    # Auto-detect configuration
#   sbatch --export=TEST_MODE=1PPN shmem_tests_auto.slurm      # Force 1 PPN mode
#   sbatch --export=TEST_MODE=MAXPPN shmem_tests_auto.slurm    # Force MAX PPN mode
#   sbatch --export=TEST_MODE=BOTH shmem_tests_auto.slurm      # Run both configurations

echo "=================================================="
echo "SHMEM Comprehensive Tests - Adaptive Configuration"
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

# Determine test mode
if [ -z "$TEST_MODE" ]; then
    # Auto-detect based on SLURM configuration
    if [ "$SLURM_NTASKS_PER_NODE" -eq 1 ]; then
        TEST_MODE="1PPN"
        echo "Auto-detected mode: 1 PPN (1 process per node)"
    elif [ "$SLURM_NTASKS_PER_NODE" -ge 8 ]; then
        TEST_MODE="MAXPPN"
        echo "Auto-detected mode: MAX PPN ($SLURM_NTASKS_PER_NODE processes per node)"
    else
        TEST_MODE="BOTH"
        echo "Auto-detected mode: BOTH (will run both 1 PPN and current configuration)"
    fi
else
    echo "User-specified mode: $TEST_MODE"
fi

echo "Test mode: $TEST_MODE"
echo ""

# Get system information
CPUS_PER_NODE=$(scontrol show node $(scontrol show hostnames $SLURM_JOB_NODELIST | head -1) | grep CPUTot | awk -F= '{print $2}' | awk '{print $1}')
MEMORY_PER_NODE=$(scontrol show node $(scontrol show hostnames $SLURM_JOB_NODELIST | head -1) | grep RealMemory | awk -F= '{print $2}' | awk '{print $1}')

echo "System Information:"
echo "- CPUs per node: ${CPUS_PER_NODE:-Unknown}"
echo "- Memory per node: ${MEMORY_PER_NODE:-Unknown} MB"
echo "- Network: $(scontrol show node $(scontrol show hostnames $SLURM_JOB_NODELIST | head -1) | grep Features | awk -F= '{print $2}' || echo 'Unknown')"
echo ""

# Load necessary modules (adjust for your system)
# module load openmpi/4.1.0
# module load shmem

# Set up environment
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

# Create main results directory
MAIN_RESULTS_DIR="results_auto_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $MAIN_RESULTS_DIR

# Save job information
cat > $MAIN_RESULTS_DIR/job_info.txt << EOF
SLURM Job Information - Adaptive Configuration
==============================================
Job ID: $SLURM_JOB_ID
Job Name: $SLURM_JOB_NAME
Nodes: $SLURM_JOB_NUM_NODES
Tasks per Node: $SLURM_NTASKS_PER_NODE
Total Tasks: $SLURM_NTASKS
CPUs per Task: $SLURM_CPUS_PER_TASK
CPUs per Node: ${CPUS_PER_NODE:-Unknown}
Memory per Node: ${MEMORY_PER_NODE:-Unknown} MB
Partition: $SLURM_JOB_PARTITION
Start Time: $(date)
Submit Directory: $SLURM_SUBMIT_DIR
Node List: $SLURM_JOB_NODELIST

Test Mode: $TEST_MODE
Purpose: Adaptive SHMEM performance testing
EOF

# Function to run tests with a specific configuration
run_test_configuration() {
    local config_name="$1"
    local pe_count="$2"
    local max_size="$3"
    local iterations="$4"
    local description="$5"
    
    echo "=================================================="
    echo "Running $config_name Configuration"
    echo "=================================================="
    echo "Description: $description"
    echo "PE Count: $pe_count"
    echo "Max Size: $max_size bytes"
    echo "Iterations: $iterations"
    echo ""
    
    # Create configuration-specific results directory
    CONFIG_RESULTS_DIR="$MAIN_RESULTS_DIR/$config_name"
    mkdir -p $CONFIG_RESULTS_DIR
    
    # Function to run a single test
    run_single_test() {
        local test_name="$1"
        local executable="$2"
        local args="$3"
        local output_file="$CONFIG_RESULTS_DIR/${test_name}.out"
        local error_file="$CONFIG_RESULTS_DIR/${test_name}.err"
        
        echo "Running $test_name..."
        echo "Command: mpirun -np $pe_count $executable $args"
        
        if timeout 900 mpirun -np $pe_count $executable $args > $output_file 2> $error_file; then
            echo "✓ $test_name completed successfully"
            # Show brief results
            echo "  Results preview:"
            tail -3 $output_file | sed 's/^/    /'
        else
            echo "✗ $test_name failed"
            if [ -s $error_file ]; then
                echo "  Error output:"
                tail -3 $error_file | sed 's/^/    /'
            fi
        fi
        echo ""
    }
    
    # Run all test categories
    echo "Point-to-Point Tests:"
    echo "--------------------"
    run_single_test "put_benchmark" "./point_to_point/shmem_put_bench" "--max-size $max_size --iterations $iterations"
    run_single_test "get_benchmark" "./point_to_point/shmem_get_bench" "--max-size $max_size --iterations $iterations"
    
    echo "Atomic Operations Tests:"
    echo "-----------------------"
    atomic_iterations=$((iterations / 2))
    run_single_test "atomic_benchmark" "./atomic/shmem_atomic_bench" "--iterations $atomic_iterations"
    
    echo "Collective Operations Tests:"
    echo "---------------------------"
    collective_size=$((max_size / 4))
    collective_iterations=$((iterations / 10))
    run_single_test "collective_benchmark" "./collective/shmem_collective_bench" "--max-size $collective_size --iterations $collective_iterations"
    
    # Generate configuration summary
    local summary_file="$CONFIG_RESULTS_DIR/summary.txt"
    cat > $summary_file << EOF
SHMEM Tests Summary - $config_name Configuration
===============================================
Job ID: $SLURM_JOB_ID
Configuration: $config_name
Description: $description
Completion Time: $(date)

Test Parameters:
- PE Count: $pe_count
- Max Size: $max_size bytes
- Iterations: $iterations

Test Results:
EOF
    
    # Check test results
    local total_tests=0
    local passed_tests=0
    
    for result_file in $CONFIG_RESULTS_DIR/*.out; do
        if [ -f "$result_file" ]; then
            total_tests=$((total_tests + 1))
            test_name=$(basename "$result_file" .out)
            
            if [ -s "$result_file" ] && ! grep -q "ERROR\|FAIL" "$result_file"; then
                passed_tests=$((passed_tests + 1))
                echo "✓ $test_name: PASSED" >> $summary_file
            else
                echo "✗ $test_name: FAILED" >> $summary_file
            fi
        fi
    done
    
    cat >> $summary_file << EOF

Summary:
- Total Tests: $total_tests
- Passed: $passed_tests  
- Failed: $((total_tests - passed_tests))
- Success Rate: $(( passed_tests * 100 / total_tests ))%

Results Directory: $CONFIG_RESULTS_DIR
EOF
    
    echo ""
    echo "$config_name Configuration Summary:"
    cat $summary_file
    echo ""
}

# Run tests based on the determined mode
case "$TEST_MODE" in
    "1PPN")
        # Run 1 PPN configuration
        pe_count=$SLURM_JOB_NUM_NODES
        max_size=1048576  # 1MB
        iterations=10000
        run_test_configuration "1PPN" $pe_count $max_size $iterations "Inter-node communication focus (1 process per node)"
        ;;
    
    "MAXPPN")
        # Run MAX PPN configuration
        pe_count=$SLURM_NTASKS
        max_size=262144   # 256KB
        iterations=5000
        run_test_configuration "MAXPPN" $pe_count $max_size $iterations "Mixed intra-node and inter-node communication ($SLURM_NTASKS_PER_NODE processes per node)"
        ;;
    
    "BOTH")
        # Run both configurations for comparison
        echo "Running both 1 PPN and current configuration for comparison..."
        echo ""
        
        # First run 1 PPN
        pe_count=$SLURM_JOB_NUM_NODES
        max_size=1048576
        iterations=8000
        run_test_configuration "1PPN" $pe_count $max_size $iterations "Inter-node communication focus (1 process per node)"
        
        # Then run current configuration
        pe_count=$SLURM_NTASKS
        max_size=262144
        iterations=4000
        run_test_configuration "CURRENT" $pe_count $max_size $iterations "Current SLURM configuration ($SLURM_NTASKS_PER_NODE processes per node)"
        ;;
    
    *)
        echo "ERROR: Unknown test mode: $TEST_MODE"
        echo "Valid modes: 1PPN, MAXPPN, BOTH"
        exit 1
        ;;
esac

# Generate overall summary
echo "=================================================="
echo "Overall Job Summary"
echo "=================================================="

OVERALL_SUMMARY="$MAIN_RESULTS_DIR/overall_summary.txt"
cat > $OVERALL_SUMMARY << EOF
SHMEM Comprehensive Tests - Overall Summary
==========================================
Job ID: $SLURM_JOB_ID
Completion Time: $(date)
Total Runtime: $SECONDS seconds

SLURM Configuration:
- Nodes: $SLURM_JOB_NUM_NODES
- Tasks per Node: $SLURM_NTASKS_PER_NODE
- Total Tasks: $SLURM_NTASKS
- Test Mode: $TEST_MODE

System Information:
- CPUs per Node: ${CPUS_PER_NODE:-Unknown}
- Memory per Node: ${MEMORY_PER_NODE:-Unknown} MB

Test Configurations Run:
EOF

# Add summaries from each configuration
for config_dir in $MAIN_RESULTS_DIR/*/; do
    if [ -d "$config_dir" ] && [ -f "$config_dir/summary.txt" ]; then
        config_name=$(basename "$config_dir")
        echo "" >> $OVERALL_SUMMARY
        echo "$config_name Configuration:" >> $OVERALL_SUMMARY
        echo "$(grep -E "Success Rate|Total Tests" "$config_dir/summary.txt")" >> $OVERALL_SUMMARY
    fi
done

cat >> $OVERALL_SUMMARY << EOF

Performance Analysis:
- Results can be compared between configurations if BOTH mode was used
- 1 PPN tests focus on inter-node network performance
- MAX PPN tests show mixed intra-node (shared memory) and inter-node performance
- Different message sizes optimize for different communication patterns

Main Results Directory: $MAIN_RESULTS_DIR
Individual Configuration Results: $MAIN_RESULTS_DIR/*/
EOF

echo ""
cat $OVERALL_SUMMARY

# Copy results to a standard location if desired
# cp -r $MAIN_RESULTS_DIR /shared/results/ 2>/dev/null || true

echo ""
echo "=================================================="
echo "Job completed at $(date)"
echo "Total runtime: $SECONDS seconds"
echo "Main results directory: $MAIN_RESULTS_DIR"
echo "==================================================" 