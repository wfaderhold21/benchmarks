# SHMEM Comprehensive Tests - SLURM Scripts

This directory contains SLURM batch scripts for running the SHMEM comprehensive test suite on HPC clusters. The scripts are designed to test different communication patterns and automatically adapt to various cluster configurations.

## Available Scripts

### 1. `shmem_tests_1ppn.slurm`
**Purpose**: Tests inter-node communication with 1 process per node (1 PPN)

**Use Case**: 
- Focuses on network fabric performance
- Ideal for testing inter-node latency and bandwidth
- Best for understanding network-bound performance

**Configuration**:
- 1 process per node across multiple nodes
- Larger message sizes (up to 1MB)
- Higher iteration counts for statistical significance

### 2. `shmem_tests_maxppn.slurm`
**Purpose**: Tests mixed communication patterns with maximum processes per node

**Use Case**:
- Tests both intra-node (shared memory) and inter-node (network) communication
- Evaluates performance with higher process density
- Shows scaling behavior with many processes

**Configuration**:
- Maximum processes per node (auto-detected or configured)
- Smaller message sizes (up to 256KB) due to higher process count
- Process binding for optimal performance

### 3. `shmem_tests_auto.slurm`
**Purpose**: Adaptive script that automatically selects configuration or runs both

**Use Case**:
- Auto-detects optimal configuration based on SLURM allocation
- Can run comparison tests between different configurations
- Best for general-purpose testing

**Modes**:
- `AUTO`: Automatically selects 1 PPN or MAX PPN based on allocation
- `1PPN`: Forces 1 process per node
- `MAXPPN`: Forces maximum processes per node  
- `BOTH`: Runs both configurations for comparison

## Quick Start

### Basic Usage

```bash
# Submit 1 PPN test (modify nodes as needed)
sbatch --nodes=4 --ntasks-per-node=1 shmem_tests_1ppn.slurm

# Submit MAX PPN test (modify based on your system)
sbatch --nodes=2 --ntasks-per-node=8 shmem_tests_maxppn.slurm

# Submit adaptive test (auto-detects configuration)
sbatch --nodes=4 --ntasks-per-node=2 shmem_tests_auto.slurm
```

### Advanced Usage

```bash
# Force specific mode in adaptive script
sbatch --export=TEST_MODE=1PPN shmem_tests_auto.slurm
sbatch --export=TEST_MODE=MAXPPN shmem_tests_auto.slurm
sbatch --export=TEST_MODE=BOTH shmem_tests_auto.slurm

# Custom resource allocation
sbatch --nodes=8 --ntasks-per-node=1 --time=01:00:00 shmem_tests_1ppn.slurm
sbatch --nodes=4 --ntasks-per-node=16 --time=00:45:00 shmem_tests_maxppn.slurm
```

## Configuration

### Customizing for Your System

Before using, modify the following in each script:

1. **SBATCH Directives** (adjust for your cluster):
```bash
#SBATCH --partition=compute     # Your partition name
#SBATCH --account=your_account  # Your account (uncomment)
#SBATCH --qos=normal           # Your QOS (uncomment)
#SBATCH --time=00:30:00        # Adjust time limit
```

2. **Environment Setup** (adjust paths):
```bash
# Load modules for your system
module load openmpi/4.1.0
module load shmem

# Or set paths directly
export PATH=/path/to/shmem/bin:$PATH
export LD_LIBRARY_PATH=/path/to/shmem/lib:$LD_LIBRARY_PATH
```

3. **Test Parameters** (optional tuning):
```bash
MAX_SIZE=1048576    # Maximum message size
ITERATIONS=10000    # Number of iterations
```

## Understanding Results

### Output Files

Each job generates several output files:

- `shmem_tests_*_<jobid>.out`: SLURM stdout
- `shmem_tests_*_<jobid>.err`: SLURM stderr  
- `results_*_<jobid>_<timestamp>/`: Results directory containing:
  - Individual test outputs (`.out` files)
  - Job information (`job_info.txt`)
  - Summary report (`summary_*.txt`)
  - Communication analysis (for MAX PPN tests)

### Result Structure

```
results_<config>_<jobid>_<timestamp>/
├── job_info.txt                    # SLURM job details
├── summary_<config>.txt            # Test summary
├── put_benchmark_<config>.out      # PUT test results
├── get_benchmark_<config>.out      # GET test results  
├── atomic_benchmark_<config>.out   # Atomic operations results
├── collective_benchmark_<config>.out # Collective operations results
└── communication_analysis.txt      # Communication pattern analysis (MAX PPN)
```

### Performance Metrics

Each test reports:
- **Latency**: Average time per operation (microseconds)
- **Bandwidth**: Data throughput (MB/s)  
- **Message Rate**: Operations per second (millions)

### Interpreting Results

**1 PPN Configuration**:
- Shows pure inter-node network performance
- Higher latencies, network-limited bandwidth
- Good for understanding network fabric capabilities

**MAX PPN Configuration**:
- Shows mixed intra-node + inter-node performance
- Lower latencies for intra-node communication
- Higher aggregate bandwidth but more contention

## Test Categories

### Point-to-Point Tests
- **PUT Operations**: `shmem_putmem` latency, bandwidth, message rate
- **GET Operations**: `shmem_getmem` latency, bandwidth, message rate

### Atomic Operations
- `atomic_fetch_add`, `atomic_add`
- `atomic_compare_swap`, `atomic_swap`
- `atomic_fetch`, `atomic_inc`
- Contention testing with multiple processes

### Collective Operations
- `alltoall64`: All-to-all data exchange
- `collect64`, `fcollect64`: Data collection operations
- `broadcast64`: Data broadcast
- `long_sum_to_all`: Reduction operations
- `barrier_all`: Synchronization latency

## Troubleshooting

### Common Issues

1. **Build Failures**:
```
ERROR: Failed to build tests
```
- Check that SHMEM compiler (`shmemcc`) is available
- Verify environment variables are set correctly
- Ensure you're in the correct directory

2. **Runtime Failures**:
```
mpirun: command not found
```
- Load appropriate modules or set PATH
- Check SHMEM launcher availability

3. **Memory Issues**:
```
Memory allocation failed
```
- Reduce message sizes or process counts
- Check available memory per node

4. **Timeout Issues**:
```
Test timeout after 600 seconds
```
- Increase timeout values in scripts
- Reduce iterations or message sizes
- Check for hanging processes

### Performance Issues

1. **Low Bandwidth**:
- Check network configuration
- Verify optimal process placement
- Consider CPU binding options

2. **High Latency**:
- Check for system load
- Verify node-to-node connectivity
- Consider process affinity settings

3. **Variable Results**:
- Ensure dedicated node access
- Run multiple iterations
- Check for thermal throttling

## Scaling Recommendations

### Small Scale (2-8 nodes)
```bash
# Good for initial testing
sbatch --nodes=4 --ntasks-per-node=1 shmem_tests_1ppn.slurm
sbatch --nodes=2 --ntasks-per-node=8 shmem_tests_maxppn.slurm
```

### Medium Scale (8-32 nodes)
```bash
# Balanced testing
sbatch --nodes=16 --ntasks-per-node=1 shmem_tests_1ppn.slurm
sbatch --nodes=8 --ntasks-per-node=4 shmem_tests_maxppn.slurm
```

### Large Scale (32+ nodes)
```bash
# Large-scale network testing
sbatch --nodes=64 --ntasks-per-node=1 --time=02:00:00 shmem_tests_1ppn.slurm
# Be careful with MAX PPN at large scale - may need reduced message sizes
```

## Customization Examples

### Custom Test Parameters

Create a custom script based on `shmem_tests_auto.slurm`:

```bash
# Custom small message test
MAX_SIZE=4096        # 4KB max
ITERATIONS=50000     # More iterations for small messages

# Custom large message test  
MAX_SIZE=4194304     # 4MB max
ITERATIONS=1000      # Fewer iterations for large messages
```

### Performance Optimization

```bash
# Add process binding for better performance
mpi_opts="--bind-to core --map-by node:PE=1"

# Use specific network interfaces
export OMPI_MCA_btl_tcp_if_include=ib0

# Set SHMEM-specific environment variables
export SHMEM_SYMMETRIC_SIZE=1GB
```

## Integration with Workflow Systems

### Slurm Job Arrays

```bash
#!/bin/bash
#SBATCH --array=1-5%2    # Run 5 jobs, max 2 concurrent

# Test different node counts
NODES_ARRAY=(2 4 8 16 32)
NODES=${NODES_ARRAY[$((SLURM_ARRAY_TASK_ID-1))]}

sbatch --nodes=$NODES --ntasks-per-node=1 shmem_tests_1ppn.slurm
```

### Dependency Chains

```bash
# Submit comparison jobs
JOB1=$(sbatch --parsable shmem_tests_1ppn.slurm)
JOB2=$(sbatch --parsable shmem_tests_maxppn.slurm)

# Submit analysis job after both complete
sbatch --dependency=afterok:$JOB1:$JOB2 analysis_script.slurm
```

## Best Practices

1. **Resource Allocation**:
   - Request appropriate time limits (30-60 minutes typical)
   - Use dedicated nodes when possible
   - Consider memory requirements for large message sizes

2. **Test Design**:
   - Run multiple trials for statistical significance
   - Compare 1 PPN vs MAX PPN for performance analysis
   - Use appropriate message sizes for your application

3. **Result Management**:
   - Archive results with descriptive names
   - Compare results across different systems/configurations
   - Monitor for performance regressions

4. **Cluster Etiquette**:
   - Don't monopolize resources unnecessarily
   - Use appropriate partition and QOS settings
   - Clean up temporary files

## Support

For issues with:
- **SLURM scripts**: Check cluster documentation and modify SBATCH directives
- **SHMEM tests**: See main README.md in parent directory
- **Performance issues**: Consult system administrators or HPC support 