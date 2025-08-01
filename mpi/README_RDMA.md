# RDMA Alltoall Benchmark

This benchmark implements an alltoall collective operation using RDMA verbs for data transfer and MPI for synchronization.

## Features

- Uses RDMA write operations for data transfer
- Uses MPI_Barrier for synchronization between iterations
- Benchmarks message sizes from 8 bytes to 1 MB
- Measures bandwidth, latency, and timing statistics
- Supports multiple MPI ranks

## Requirements

- MPI implementation (OpenMPI, MPICH, etc.)
- RDMA-capable network (InfiniBand, RoCE, iWARP)
- libibverbs development library
- GCC or compatible C compiler

## Building

```bash
make rdma_alltoall_bench
```

## Running

```bash
# Run with 4 ranks
mpirun -np 4 ./rdma_alltoall_bench

# Run with specific number of ranks
mpirun -np 8 ./rdma_alltoall_bench
```

## Output

The benchmark outputs performance metrics for each message size:

- **Message Size**: Size of each message in bytes
- **Total Size**: Total data transferred (message_size × num_ranks²)
- **Bandwidth (MB/s)**: Achieved bandwidth
- **Avg Latency (us)**: Average latency in microseconds
- **Min Latency (us)**: Minimum latency in microseconds
- **Max Latency (us)**: Maximum latency in microseconds

## Implementation Details

### RDMA Setup
1. **Device Discovery**: Finds available InfiniBand devices
2. **Context Creation**: Opens device context and allocates protection domain
3. **Memory Registration**: Registers memory regions for RDMA operations
4. **Queue Pair Creation**: Creates completion queues and queue pairs
5. **Connection Setup**: Exchanges QP information and memory keys (rkeys) between ranks
6. **Memory Key Exchange**: Each rank shares its memory region's remote key (rkey) and buffer address

### Alltoall Algorithm
1. Each rank sends its data to all other ranks using RDMA write operations
2. Remote memory access is performed using exchanged memory keys (rkeys)
3. Local data is copied directly to the receive buffer
4. Remote data is written to the RDMA buffer and then copied to the receive buffer
5. MPI_Barrier ensures synchronization between iterations

### Performance Measurement
- Warmup iterations to stabilize performance
- Multiple benchmark iterations for statistical accuracy
- Timing measurements using MPI_Wtime()
- Bandwidth calculation based on total data transferred

## Configuration

You can modify the following constants in the source code:

- `NR_ITER`: Number of benchmark iterations (default: 110)
- `SKIP`: Number of warmup iterations (default: 10)
- `MAX_MSG_SIZE`: Maximum message size (default: 1MB)
- `MIN_MSG_SIZE`: Minimum message size (default: 8 bytes)

## Troubleshooting

### Common Issues

1. **No IB devices found**: Ensure InfiniBand drivers are loaded and devices are available
2. **Permission denied**: Run with appropriate permissions or use sudo
3. **Connection failed**: Check network configuration and firewall settings
4. **Memory allocation failed**: Reduce MAX_MSG_SIZE or number of ranks

### Debugging

Enable verbose output by modifying the source code to add debug prints in the RDMA setup functions.

## Performance Tips

1. **Message Size**: Larger messages generally achieve higher bandwidth
2. **Number of Ranks**: More ranks increase total data transfer but may reduce per-rank bandwidth
3. **Network Configuration**: Ensure optimal network settings for your InfiniBand fabric
4. **CPU Affinity**: Pin processes to specific CPU cores for better performance

## Scalability Considerations

### No Artificial Rank Limits
The benchmark supports **any number of ranks** - there are no artificial limits. The only constraints are:

1. **System Resources**: Available memory, CPU cores, and network bandwidth
2. **InfiniBand Hardware**: Number of available ports and QP capacity
3. **MPI Implementation**: Maximum number of processes supported by your MPI

### Practical Limits
While there's no code limit, practical considerations include:

- **Memory Usage**: Each rank allocates `num_ranks * MAX_MSG_SIZE` bytes for buffers
- **QP Creation**: Each rank creates `(num_ranks - 1)` queue pairs
- **Network Bandwidth**: Total bandwidth scales with `num_ranks²`
- **Setup Time**: QP creation and connection time increases with rank count

### Scaling Recommendations
- **Small Scale (2-16 ranks)**: Direct testing, minimal setup overhead
- **Medium Scale (16-64 ranks)**: Monitor memory usage and QP creation time
- **Large Scale (64+ ranks)**: Consider memory-efficient algorithms and QP pooling
- **Very Large Scale (1000+ ranks)**: May require specialized RDMA implementations

## Comparison with MPI Alltoall

This benchmark can be compared with standard MPI alltoall implementations to measure the performance benefits of RDMA-based communication. 