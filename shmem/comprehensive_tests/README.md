# SHMEM Comprehensive Benchmark Suite

This directory contains a comprehensive suite of SHMEM (Symmetric Hierarchical Memory) benchmarks designed to measure the performance characteristics of various SHMEM operations including point-to-point communication, atomic operations, and collective operations.

## Overview

The benchmark suite is organized into the following categories:

- **Point-to-Point Operations**: `shmem_put` and `shmem_get` operations
- **Atomic Operations**: Various atomic operations including fetch_add, compare_swap, etc.
- **Collective Operations**: Alltoall, collect, fcollect, broadcast, and reduce operations

## Directory Structure

```
comprehensive_tests/
├── common/
│   └── common.h              # Shared utilities and helper functions
├── point_to_point/
│   ├── shmem_put_bench.c     # PUT operation benchmarks
│   └── shmem_get_bench.c     # GET operation benchmarks
├── atomic/
│   └── shmem_atomic_bench.c  # Atomic operations benchmarks
├── collective/
│   └── shmem_collective_bench.c # Collective operations benchmarks
├── slurm-scripts/
│   ├── shmem_tests_1ppn.slurm    # SLURM script for 1 PPN testing
│   ├── shmem_tests_maxppn.slurm  # SLURM script for MAX PPN testing
│   ├── shmem_tests_auto.slurm    # Adaptive SLURM script
│   └── README.md             # SLURM scripts documentation
├── Makefile                  # Main build file
├── run_all_tests.sh         # Comprehensive test runner script
└── README.md                # This file
```

## Features

### Measurements
Each benchmark measures:
- **Latency**: Average time per operation in microseconds
- **Bandwidth**: Data throughput in MB/s
- **Message Rate**: Operations per second (for small messages)

### Test Categories

#### 1. Point-to-Point Tests
- **shmem_put_bench**: Tests `shmem_putmem` operations
  - Latency and bandwidth across message sizes (1B to 1MB)
  - Message rate for small 8-byte messages
  - Data validation
  
- **shmem_get_bench**: Tests `shmem_getmem` operations
  - Latency and bandwidth across message sizes
  - Message rate measurements
  - Bi-directional testing capabilities

#### 2. Atomic Operations Tests
- **shmem_atomic_bench**: Tests various atomic operations
  - `atomic_fetch_add`, `atomic_add`
  - `atomic_compare_swap`, `atomic_swap`
  - `atomic_fetch`, `atomic_inc`
  - Contention testing with multiple PEs

#### 3. Collective Operations Tests
- **shmem_collective_bench**: Tests collective operations
  - `alltoall64`: All-to-all data exchange
  - `collect64`: Data collection
  - `fcollect64`: Fast collection
  - `broadcast64`: Data broadcast
  - `long_sum_to_all`: Reduction operations
  - `barrier_all`: Synchronization latency

## Building the Tests

### Prerequisites
- SHMEM implementation (OpenSHMEM, Cray SHMEM, etc.)
- `shmemcc` compiler wrapper
- SHMEM launcher (`shmrun`, `mpirun`, or `aprun`)

### Build Commands
```bash
# Build all tests
make all

# Build specific categories
make point_to_point
make atomic
make collective

# Get help
make help
```

### Build Options
You can customize the build with environment variables:
```bash
# Use custom compiler flags
make CFLAGS="-O3 -DWITH_HINTS" all

# Use different compiler
make CC=oshcc all
```

## Running the Tests

### Quick Start
```bash
# Make the test script executable
chmod +x run_all_tests.sh

# Run all tests with default settings
./run_all_tests.sh

# Run quick validation tests only
./run_all_tests.sh --validation-only

# Run with custom settings
./run_all_tests.sh --pe-counts "2 4 8 16" --max-size 1048576
```

### Manual Test Execution

#### Individual Tests
```bash
# PUT benchmark with 4 PEs
shmrun -np 4 point_to_point/shmem_put_bench

# GET benchmark with custom parameters
shmrun -np 8 point_to_point/shmem_get_bench --max-size 65536 --iterations 10000

# Atomic operations benchmark
shmrun -np 4 atomic/shmem_atomic_bench

# Collective operations benchmark
shmrun -np 8 collective/shmem_collective_bench
```

#### Using Makefile Targets
```bash
# Run individual test categories
make test-put
make test-get
make test-atomic
make test-collective

# Run performance tests with multiple PE counts
make perf-test

# Run validation tests
make validation-test
```

#### Using SLURM Scripts (HPC Clusters)
```bash
# Submit 1 PPN test (inter-node communication focus)
sbatch --nodes=4 --ntasks-per-node=1 slurm-scripts/shmem_tests_1ppn.slurm

# Submit MAX PPN test (mixed communication patterns)
sbatch --nodes=2 --ntasks-per-node=8 slurm-scripts/shmem_tests_maxppn.slurm

# Submit adaptive test (auto-detects configuration)
sbatch --nodes=4 --ntasks-per-node=2 slurm-scripts/shmem_tests_auto.slurm

# Force specific mode in adaptive script
sbatch --export=TEST_MODE=BOTH slurm-scripts/shmem_tests_auto.slurm
```

See `slurm-scripts/README.md` for detailed SLURM usage documentation.

## Command Line Options

### Test Programs
All test programs support common command line options:

- `--max-size SIZE`: Maximum message size to test (default: 1MB for point-to-point, 64KB for collectives)
- `--iterations NUM`: Number of iterations per test (default: 100,000)
- `--warmup NUM`: Number of warmup iterations (default: 1,000)
- `--no-validate`: Disable data validation

### Test Runner Script
The `run_all_tests.sh` script supports:

- `-p, --pe-counts "2 4 8"`: PE counts to test
- `-s, --max-size 65536`: Maximum message size
- `-i, --iterations 10000`: Number of iterations
- `-l, --launcher shmrun`: SHMEM launcher command
- `-v, --validation-only`: Run only validation tests
- `-q, --quick`: Run quick tests (reduced iterations)
- `-h, --help`: Show help message

## Understanding the Output

### Sample Output Format
```
===============================================
 SHMEM PUT LATENCY/BANDWIDTH BENCHMARK RESULTS
===============================================
Size         Latency(us)     Bandwidth(MB/s) Msg Rate(M/s)
---------------------------------------------------------------
1 B          0.52            1.83            1.92
2 B          0.53            3.60            1.89
4 B          0.54            7.08            1.85
...
1.00 MB      45.23           23.15           0.02
```

### Key Metrics
- **Latency**: Lower is better (microseconds per operation)
- **Bandwidth**: Higher is better (MB/s for large messages)
- **Message Rate**: Higher is better (million messages per second for small messages)

## Performance Tuning

### Compile-time Options
```bash
# Enable memory hints (if supported)
make CFLAGS="-DWITH_HINTS" all

# Optimize for specific architecture
make CFLAGS="-O3 -march=native" all
```

### Runtime Considerations
1. **PE Placement**: Ensure PEs are optimally placed across nodes
2. **Memory Affinity**: Use NUMA-aware PE placement
3. **Network Configuration**: Optimize interconnect settings
4. **System Load**: Run on dedicated systems for consistent results

## Validation and Correctness

All benchmarks include built-in validation:
- **Data Integrity**: Verify that transferred data matches expected values
- **Atomic Correctness**: Check atomic operation results
- **Collective Validation**: Ensure collective operations produce correct results

Validation can be disabled with `--no-validate` for pure performance testing.

## Results Analysis

### Automated Reporting
The test runner generates:
- Individual result files for each test and PE count
- System information file
- Summary report with pass/fail status
- Performance highlights

### Result Files Location
Results are saved in timestamped directories:
```
results_YYYYMMDD_HHMMSS/
├── system_info.txt
├── summary_report.txt
├── put_bench_4pe.out
├── get_bench_4pe.out
├── atomic_bench_4pe.out
├── collective_bench_4pe.out
└── ...
```

## Troubleshooting

### Common Issues

1. **Compiler Not Found**
   ```
   Error: shmemcc not found
   Solution: Ensure SHMEM is properly installed and in PATH
   ```

2. **Launcher Issues**
   ```
   Error: shmrun not found
   Solution: Use alternative launcher with -l option
   ```

3. **Memory Allocation Failures**
   ```
   Error: Memory allocation failed
   Solution: Reduce message sizes or increase available memory
   ```

4. **Test Timeouts**
   ```
   Error: Test timeout after 300 seconds
   Solution: Reduce iterations or increase timeout in script
   ```

### Performance Issues
- **Low Bandwidth**: Check network configuration and PE placement
- **High Latency**: Verify optimal PE-to-node mapping
- **Variable Results**: Ensure system is not under load

## Extending the Test Suite

### Adding New Tests
1. Create new test file in appropriate directory
2. Include `../common/common.h`
3. Use common utility functions
4. Add build target to Makefile
5. Update test runner script if needed

### Custom Benchmarks
The common utilities provide:
- Timing functions (`TIME()` macro)
- Result formatting (`print_benchmark_result()`)
- Memory allocation helpers (`shmem_malloc_aligned()`)
- Data validation functions

## Contributing

When contributing new tests or improvements:
1. Follow the existing code structure and style
2. Include appropriate validation
3. Add documentation for new features
4. Test with multiple PE counts and message sizes

## References

- [OpenSHMEM Specification](http://openshmem.org/)
- [SHMEM Programming Model](https://www.openmp.org/wp-content/uploads/shmem-1.4.pdf)
- [Performance Analysis Best Practices](http://www.openshmem.org/site/sites/default/site_files/OpenSHMEM-SWG-1.4.pdf)

## License

This benchmark suite is provided as-is for performance analysis and testing purposes. 