# MPI and RDMA Alltoall Benchmarks

This directory contains various implementations of alltoall collective operation benchmarks, including RDMA-based versions using RDMA verbs and MPI for synchronization.

## Available Benchmarks

### 1. `mpi_bench_a2a` - Standard MPI Alltoall
- Uses standard MPI alltoall collective
- Includes UCC (Unified Communication Collective) support
- Compiles with: `make mpi_bench_a2a`

### 2. `ucc_bench_a2a` - UCC-based Alltoall
- Uses UCC library for alltoall operations
- Requires UCC installation
- Compiles with: `make ucc_bench_a2a`

### 3. `rdma_alltoall_bench` - Full RDMA Implementation
- Complete RDMA verbs implementation
- Requires InfiniBand hardware and libibverbs
- Uses MPI for synchronization
- Compiles with: `make rdma_alltoall_bench`

### 4. `rdma_alltoall_simple` - RDMA with Hardware Dependencies
- Simplified RDMA implementation
- Requires libibverbs library
- Compiles with: `make rdma_alltoall_simple`

### 5. `rdma_alltoall_demo` - RDMA Concept Demonstration
- Demonstrates RDMA alltoall concept without hardware requirements
- Simulates RDMA operations for educational purposes
- Compiles with: `make rdma_alltoall_demo`

## Building All Benchmarks

```bash
# Build all benchmarks
make all

# Build specific benchmark
make rdma_alltoall_demo
make mpi_bench_a2a
make ucc_bench_a2a
```

## Running the Benchmarks

### RDMA Demo (Works without special hardware)
```bash
./rdma_alltoall_demo
```

### MPI Benchmarks (Requires MPI installation)
```bash
# Run with 4 ranks
mpirun -np 4 ./mpi_bench_a2a

# Run with 8 ranks
mpirun -np 8 ./ucc_bench_a2a
```

### Full RDMA Benchmark (Requires InfiniBand hardware)
```bash
# Run with 4 ranks
mpirun -np 4 ./rdma_alltoall_bench
```

## RDMA Alltoall Implementation Details

### Key Features
- **RDMA Write Operations**: Uses RDMA write for data transfer
- **MPI Synchronization**: Uses MPI_Barrier for synchronization between iterations
- **Multiple Message Sizes**: Benchmarks from 8 bytes to 1 MB
- **Performance Metrics**: Measures bandwidth, latency, and timing statistics

### Algorithm Overview
1. **Setup Phase**:
   - Initialize RDMA context and device
   - Allocate and register memory regions
   - Create queue pairs and completion queues
   - Exchange QP information between ranks
   - Establish connections

2. **Alltoall Operation**:
   - Each rank sends data to all other ranks using RDMA write
   - Local data is copied directly
   - Remote data is written to RDMA buffers
   - Data is copied from RDMA buffers to receive buffers

3. **Synchronization**:
   - MPI_Barrier ensures all ranks are synchronized
   - Used between iterations for accurate timing

### Performance Measurement
- **Warmup Iterations**: 10 iterations to stabilize performance
- **Benchmark Iterations**: 110 iterations for statistical accuracy
- **Timing**: Uses high-resolution timers for microsecond precision
- **Statistics**: Calculates min, max, and average latencies
- **Bandwidth**: Computes effective bandwidth in MB/s

## Requirements

### For Demo Version (rdma_alltoall_demo)
- GCC compiler
- Standard C libraries
- No special hardware required

### For Full RDMA Version (rdma_alltoall_bench)
- MPI implementation (OpenMPI, MPICH)
- InfiniBand hardware
- libibverbs development library
- RDMA-capable network

### For MPI Versions
- MPI implementation
- UCC library (for ucc_bench_a2a)

## Installation Dependencies

### Ubuntu/Debian
```bash
# For MPI
sudo apt install mpich libmpich-dev

# For RDMA (if hardware available)
sudo apt install libibverbs-dev

# For UCC (if needed)
# Follow UCC installation instructions
```

### CentOS/RHEL
```bash
# For MPI
sudo yum install mpich-devel

# For RDMA (if hardware available)
sudo yum install libibverbs-devel
```

## Output Format

The benchmarks output performance metrics in a tabular format:

```
Message Size  Total Size    Bandwidth (MB/s)  Avg Latency (us)  Min Latency (us)  Max Latency (us)
--------------------------------------------------------------------------------
8            128           0.15              52.34             45.12             67.89
16           256           0.31              48.67             42.34             59.12
...
1048576      16777216     3902.95           4099.46           3828.00           4272.00
```

## Configuration

You can modify the following constants in the source files:

- `NR_ITER`: Number of benchmark iterations (default: 110)
- `SKIP`: Number of warmup iterations (default: 10)
- `MAX_MSG_SIZE`: Maximum message size (default: 1MB)
- `MIN_MSG_SIZE`: Minimum message size (default: 8 bytes)

## Troubleshooting

### Common Issues

1. **"No IB devices found"**
   - Ensure InfiniBand drivers are loaded
   - Check if IB hardware is available

2. **"Permission denied"**
   - Run with appropriate permissions
   - Check device permissions

3. **"MPI not found"**
   - Install MPI implementation
   - Ensure mpicc is in PATH

4. **"libibverbs not found"**
   - Install libibverbs-dev package
   - Ensure IB hardware is available

### Debugging

Enable verbose output by modifying the source code to add debug prints in the RDMA setup functions.

## Performance Tips

1. **Message Size**: Larger messages generally achieve higher bandwidth
2. **Number of Ranks**: More ranks increase total data transfer but may reduce per-rank bandwidth
3. **Network Configuration**: Ensure optimal network settings for your InfiniBand fabric
4. **CPU Affinity**: Pin processes to specific CPU cores for better performance

## Comparison

The different versions allow you to compare:

- **Standard MPI vs RDMA**: Performance benefits of RDMA-based communication
- **UCC vs Native MPI**: Benefits of optimized collective libraries
- **Simulation vs Real Hardware**: Understanding RDMA concepts vs actual performance

## Files

- `mpi_bench_a2a.c`: Standard MPI alltoall benchmark
- `rdma_alltoall_bench.c`: Full RDMA implementation
- `rdma_alltoall_simple.c`: Simplified RDMA version
- `rdma_alltoall_demo.c`: RDMA concept demonstration
- `Makefile`: Build configuration
- `README_RDMA.md`: Detailed RDMA documentation
- `README.md`: This comprehensive guide 