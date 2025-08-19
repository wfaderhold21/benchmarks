#!/bin/bash

# SHMEM Comprehensive Benchmark Test Suite
# =========================================
# This script runs all SHMEM benchmarks with various configurations
# to provide comprehensive performance analysis.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_PE_COUNTS="2 4 8"
DEFAULT_MAX_SIZE=65536
DEFAULT_ITERATIONS=10000
SHMEM_LAUNCHER="shmrun"

# Results directory
RESULTS_DIR="results_$(date +%Y%m%d_%H%M%S)"

# Function to print colored output
print_header() {
    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN} $1${NC}"
    echo -e "${CYAN}============================================${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check if shmemcc is available
    if ! command -v shmemcc &> /dev/null; then
        print_error "shmemcc not found. Please ensure SHMEM is properly installed."
        exit 1
    fi
    print_success "shmemcc found"
    
    # Check if shmrun is available
    if ! command -v $SHMEM_LAUNCHER &> /dev/null; then
        print_warning "$SHMEM_LAUNCHER not found. Will try alternative launchers."
        
        # Try alternative launchers
        if command -v mpirun &> /dev/null; then
            SHMEM_LAUNCHER="mpirun"
            print_info "Using mpirun as launcher"
        elif command -v aprun &> /dev/null; then
            SHMEM_LAUNCHER="aprun"
            print_info "Using aprun as launcher"
        else
            print_error "No suitable SHMEM launcher found (shmrun, mpirun, aprun)"
            exit 1
        fi
    else
        print_success "$SHMEM_LAUNCHER found"
    fi
}

# Function to build all tests
build_tests() {
    print_header "Building SHMEM Benchmarks"
    
    if make clean && make all; then
        print_success "All benchmarks built successfully"
    else
        print_error "Failed to build benchmarks"
        exit 1
    fi
}

# Function to create results directory
setup_results() {
    print_header "Setting Up Results Directory"
    
    mkdir -p "$RESULTS_DIR"
    print_info "Results will be saved to: $RESULTS_DIR"
    
    # Save system information
    {
        echo "SHMEM Benchmark Results"
        echo "======================"
        echo "Date: $(date)"
        echo "Host: $(hostname)"
        echo "User: $(whoami)"
        echo "OS: $(uname -a)"
        echo ""
        echo "SHMEM Launcher: $SHMEM_LAUNCHER"
        echo "PE Counts: $DEFAULT_PE_COUNTS"
        echo "Max Size: $DEFAULT_MAX_SIZE"
        echo "Iterations: $DEFAULT_ITERATIONS"
        echo ""
    } > "$RESULTS_DIR/system_info.txt"
}

# Function to run a single benchmark
run_benchmark() {
    local test_name="$1"
    local executable="$2"
    local pe_count="$3"
    local args="$4"
    
    print_info "Running $test_name with $pe_count PEs..."
    
    local output_file="$RESULTS_DIR/${test_name}_${pe_count}pe.out"
    local error_file="$RESULTS_DIR/${test_name}_${pe_count}pe.err"
    
    if timeout --preserve-status --signal=TERM --kill-after=30s 120 $SHMEM_LAUNCHER -np $pe_count $executable $args > "$output_file" 2> "$error_file"; then
        print_success "$test_name ($pe_count PEs) completed"
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            print_error "$test_name ($pe_count PEs) TIMEOUT after 120 seconds"
            # Try to clean up any stuck processes
            pkill -f "$executable" 2>/dev/null || true
            sleep 2
            pkill -9 -f "$executable" 2>/dev/null || true
        elif [ $exit_code -eq 137 ]; then
            print_error "$test_name ($pe_count PEs) KILLED by timeout"
        else
            print_error "$test_name ($pe_count PEs) failed with exit code $exit_code"
        fi

        if [ -s "$error_file" ]; then
            print_error "Error output:"
            cat "$error_file"
        fi
        return 1
    fi
}

# Function to run point-to-point tests
run_point_to_point_tests() {
    print_header "Running Point-to-Point Tests"
    
    local args="--max-size $DEFAULT_MAX_SIZE --iterations $DEFAULT_ITERATIONS"
    
    for pe_count in $DEFAULT_PE_COUNTS; do
        run_benchmark "put_bench" "point_to_point/shmem_put_bench" "$pe_count" "$args"
        run_benchmark "get_bench" "point_to_point/shmem_get_bench" "$pe_count" "$args"
    done
}

# Function to run atomic tests
run_atomic_tests() {
    print_header "Running Atomic Operations Tests"
    
    local args="--iterations $DEFAULT_ITERATIONS"
    
    for pe_count in $DEFAULT_PE_COUNTS; do
        run_benchmark "atomic_bench" "atomic/shmem_atomic_bench" "$pe_count" "$args"
    done
}

# Function to run collective tests
run_collective_tests() {
    print_header "Running Collective Operations Tests"
    
    local collective_size=$((DEFAULT_MAX_SIZE / 4))  # Smaller size for collectives
    local collective_iterations=$((DEFAULT_ITERATIONS / 10))  # Fewer iterations for collectives
    local args="--max-size $collective_size --iterations $collective_iterations"
    
    for pe_count in $DEFAULT_PE_COUNTS; do
        run_benchmark "collective_bench" "collective/shmem_collective_bench" "$pe_count" "$args"
    done
}

# Function to run validation tests
run_validation_tests() {
    print_header "Running Validation Tests"
    
    print_info "Running quick validation with 4 PEs..."
    
    local val_args="--max-size 1024 --iterations 100"
    
    # Run validation tests
    run_benchmark "put_validation" "point_to_point/shmem_put_bench" "4" "$val_args"
    run_benchmark "get_validation" "point_to_point/shmem_get_bench" "4" "$val_args"
    run_benchmark "atomic_validation" "atomic/shmem_atomic_bench" "4" "--iterations 1000"
    run_benchmark "collective_validation" "collective/shmem_collective_bench" "4" "--max-size 1024 --iterations 10"
}

# Function to generate summary report
generate_summary() {
    print_header "Generating Summary Report"
    
    local summary_file="$RESULTS_DIR/summary_report.txt"
    
    {
        echo "SHMEM Comprehensive Benchmark Summary"
        echo "====================================="
        echo "Generated: $(date)"
        echo ""
        
        echo "Test Results Overview:"
        echo "====================="
        
        # Count successful and failed tests
        local total_tests=0
        local passed_tests=0
        
        for file in "$RESULTS_DIR"/*.out; do
            if [ -f "$file" ]; then
                total_tests=$((total_tests + 1))
                if [ -s "$file" ] && ! grep -q "ERROR\|FAIL" "$file"; then
                    passed_tests=$((passed_tests + 1))
                fi
            fi
        done
        
        echo "Total Tests: $total_tests"
        echo "Passed: $passed_tests"
        echo "Failed: $((total_tests - passed_tests))"
        echo ""
        
        if [ $passed_tests -eq $total_tests ]; then
            echo "✅ All tests passed successfully!"
        else
            echo "❌ Some tests failed. Check individual result files for details."
        fi
        
        echo ""
        echo "Performance Highlights:"
        echo "======================"
        
        # Extract some key performance metrics (simplified)
        if [ -f "$RESULTS_DIR/put_bench_4pe.out" ]; then
            echo "PUT Performance (4 PEs):"
            grep -E "1\.00 MB|Bandwidth" "$RESULTS_DIR/put_bench_4pe.out" | tail -2 || true
        fi
        
        if [ -f "$RESULTS_DIR/collective_bench_4pe.out" ]; then
            echo ""
            echo "Collective Performance (4 PEs):"
            grep -E "ALLTOALL|1\.00 KB" "$RESULTS_DIR/collective_bench_4pe.out" | tail -2 || true
        fi
        
        echo ""
        echo "Files in results directory:"
        ls -la "$RESULTS_DIR/"
        
    } > "$summary_file"
    
    print_success "Summary report generated: $summary_file"
    
    # Display summary
    echo ""
    cat "$summary_file"
}

# Function to cleanup on exit
cleanup() {
    print_info "Cleaning up..."
    # Add any cleanup tasks here if needed
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -p, --pe-counts COUNTS    PE counts to test (default: \"$DEFAULT_PE_COUNTS\")"
    echo "  -s, --max-size SIZE       Maximum message size (default: $DEFAULT_MAX_SIZE)"
    echo "  -i, --iterations NUM      Number of iterations (default: $DEFAULT_ITERATIONS)"
    echo "  -l, --launcher LAUNCHER   SHMEM launcher (default: $SHMEM_LAUNCHER)"
    echo "  -v, --validation-only     Run only validation tests"
    echo "  -q, --quick               Run quick tests (smaller sizes, fewer iterations)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all tests with default settings"
    echo "  $0 -p \"2 4 8 16\" -s 1048576         # Test with specific PE counts and larger size"
    echo "  $0 -v                                # Run only validation tests"
    echo "  $0 -q                                # Run quick tests"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--pe-counts)
            DEFAULT_PE_COUNTS="$2"
            shift 2
            ;;
        -s|--max-size)
            DEFAULT_MAX_SIZE="$2"
            shift 2
            ;;
        -i|--iterations)
            DEFAULT_ITERATIONS="$2"
            shift 2
            ;;
        -l|--launcher)
            SHMEM_LAUNCHER="$2"
            shift 2
            ;;
        -v|--validation-only)
            VALIDATION_ONLY=1
            shift
            ;;
        -q|--quick)
            DEFAULT_MAX_SIZE=4096
            DEFAULT_ITERATIONS=1000
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Enhanced cleanup function
cleanup() {
    print_info "Cleaning up..."

    # Kill any remaining SHMEM processes
    pkill -f "shmem.*bench" 2>/dev/null || true
    pkill -f "shmrun" 2>/dev/null || true
    pkill -f "mpirun.*shmem" 2>/dev/null || true

    # Wait a bit then force kill if needed
    sleep 2
    pkill -9 -f "shmem.*bench" 2>/dev/null || true
    pkill -9 -f "shmrun" 2>/dev/null || true
    pkill -9 -f "mpirun.*shmem" 2>/dev/null || true

    print_info "Cleanup complete"
}

# Enhanced signal handling
trap cleanup EXIT
trap 'print_error "Interrupted by user"; cleanup; exit 130' INT
trap 'print_error "Terminated"; cleanup; exit 143' TERM

# Main execution
main() {
    print_header "SHMEM Comprehensive Benchmark Suite"
    
    check_prerequisites
    build_tests
    setup_results
    
    if [ "${VALIDATION_ONLY:-0}" -eq 1 ]; then
        run_validation_tests
    else
        run_point_to_point_tests
        run_atomic_tests
        run_collective_tests
        run_validation_tests
    fi
    
    generate_summary
    
    print_success "All tests completed! Results saved in: $RESULTS_DIR"
}

# Run main function
main "$@" 
