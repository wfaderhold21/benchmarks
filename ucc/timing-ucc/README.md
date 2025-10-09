# UCC Context/Team Creation Benchmark

A benchmark tool to measure the time taken for UCC (Unified Collective Communication) context and team creation operations using MPI.

## Features

### Statistical Analysis
- **Average, Min, Max** timing measurements
- **Standard deviation** and **variance** calculations
- Per-iteration timing display
- Warm-up iterations to avoid cold-start effects

### Transport Layer Information
**Dynamically detects actual loaded transport layers at runtime!**

Uses `dl_iterate_phdr()` to inspect loaded shared libraries and identify:

**UCC Transport Layers (from `libucc_tl_*.so`):**
- Detects which UCC TLs are **actually loaded** at runtime
- Shows: `ucp` (UCX), `nccl` (NVIDIA), `sharp` (Mellanox), `self`, `shm`, `cuda`, `rocm`
- Provides descriptions for each loaded TL

**UCX Transports (from `libuct_*.so`):**
- Detects which UCX transport libraries are **actually loaded** at runtime
- Shows: InfiniBand transports, TCP, shared memory, CUDA, etc.
- Distinguishes between requested (env vars) vs actually loaded

**UCC Context Attributes (via `ucc_context_get_attr`):**
- Context address length
- Work buffer size
- Thread mode

**Environment Variables Checked:**
- `UCC_TLS` / `UCC_CLS` - UCC transport/collective layer selection
- `UCX_TLS` - UCX transport layer selection
- `UCX_NET_DEVICES` - Network device selection
- `UCC_TL_UCP_TUNE` / `UCC_CL_BASIC_TUNE` - Tuning parameters
- `NCCL_*` - NCCL configuration (if using NCCL TL)

**Key Advantage:** Unlike environment variable checking or log parsing, this approach shows exactly which transport libraries are loaded into the process address space, giving you the ground truth about what's actually being used.

### Code Quality
- Pure MPI implementation (no SHMEM dependencies)
- Proper memory cleanup
- Clear error handling
- Well-structured output

## Building

### Prerequisites
- MPI implementation (OpenMPI, MPICH, etc.)
- UCC library
- UCX library (typically required by UCC)

### Probe Available Transports

Before building, you can check what transports are available on your system:

```bash
./probe_transports.sh
```

This script will:
- Detect UCX and UCC installations
- List available transport layers
- Check for InfiniBand and GPU devices
- Provide recommended configurations for your hardware

### Compilation

```bash
make
```

Or manually:
```bash
mpicc -Wall -O2 -g -o time-ucc time-ucc.c -lucc -lucs -lopen-pal -lm
```

**Note:** You may need to adjust library paths and dependencies based on your installation:
- `-lucs` - UCX service library (required by UCC)
- `-lopen-pal` - Open MPI runtime library (if using Open MPI)
- Add `-I` and `-L` flags if libraries are not in standard locations

## Usage

### Basic Run
```bash
mpirun -np 4 ./time-ucc
```

### With UCX Transport Configuration
```bash
# Use specific transports
export UCX_TLS=rc,self,sm
export UCX_NET_DEVICES=mlx5_0:1
mpirun -np 4 ./time-ucc
```

### With UCC Configuration
```bash
# Enable specific UCC TLs
export UCC_TLS=ucp,sharp
export UCC_TL_UCP_TUNE=inf
mpirun -np 4 ./time-ucc

# Force only UCX-based TL
export UCC_TLS=ucp
mpirun -np 4 ./time-ucc

# Use NCCL TL (if available and using GPUs)
export UCC_TLS=nccl,ucp
mpirun -np 4 ./time-ucc
```

### Testing Different UCX Transports
```bash
# InfiniBand only (fastest)
export UCX_TLS=rc,sm,self
mpirun -np 4 ./time-ucc

# TCP/IP (compatibility mode)
export UCX_TLS=tcp,sm,self
mpirun -np 4 ./time-ucc

# Shared memory only (single node)
export UCX_TLS=sm,self
mpirun -np 4 ./time-ucc
```

## Output

The benchmark produces structured output including:

1. **Configuration Summary**
   - Number of processes
   - Iterations and skip count

2. **Context Creation Timing**
   - Per-iteration measurements
   - Statistical summary (avg, min, max, stddev, variance)

3. **Team Creation Timing**
   - Per-iteration measurements
   - Statistical summary

4. **UCC Configuration**
   - Active environment variables
   - Transport layer settings

### Example Output

```
========================================
UCC Context/Team Creation Benchmark
========================================
Processes:   4
Iterations:  10
Skip:        1
========================================

>>> Starting Context Creation Benchmark <<<
  Iteration  1:  45.23 ms
  Iteration  2:  43.12 ms
  ...

=== Context Creation Statistics ===
  Iterations:  9
  Total:       401.25 ms
  Average:     44.58 ms
  Min:         43.12 ms
  Max:         47.89 ms
  Std Dev:     1.45 ms
  Variance:    2.10 ms²

>>> Starting Team Creation Benchmark <<<
  ...

=== UCC Configuration ===
  Thread mode: 0
  Context Attributes:
    Context address length: 256 bytes
    Work buffer size: 65536 bytes

  Loaded Transport Layers (via dl_iterate_phdr):
    Found 2 loaded UCC TLs:
      - ucp (UCX Protocol - general purpose)
      - nccl (NVIDIA NCCL - GPU collectives)
    Requested via UCC_TLS: ucp,nccl

  Transport Environment Variables:
    UCC_TLS = ucp,nccl
    UCX_TLS = rc,self,sm
    UCX_NET_DEVICES = mlx5_0:1

  Loaded UCX Transports (via dl_iterate_phdr):
    Found 3 loaded UCX transports:
      - ib (InfiniBand/Mellanox)
      - sm (Shared Memory)
      - self (Loopback)
    Requested via UCX_TLS: rc,self,sm
```

### When Environment Variables Are Not Set

If you don't set any transport environment variables, the output will show:

```
=== UCC Configuration ===
  Thread mode: 0
  Context Attributes:
    Context address length: 256 bytes
    Work buffer size: 65536 bytes

  Loaded Transport Layers (via dl_iterate_phdr):
    Found 1 loaded UCC TL:
      - ucp (UCX Protocol - general purpose)
    UCC_TLS not set (using all available TLs)

  Transport Environment Variables:
    (No transport environment variables set - using UCC/UCX defaults)

  Loaded UCX Transports (via dl_iterate_phdr):
    Found 4 loaded UCX transports:
      - ib_mlx5 (InfiniBand/Mellanox)
      - tcp (TCP/IP sockets)
      - sm (Shared Memory)
      - self (Loopback)
    UCX_TLS not set (auto-detection enabled)
    Tip: Set UCX_LOG_LEVEL=INFO to see transport selection details
```

**What this means:**
- UCC/UCX will **automatically detect** available transports
- Common defaults include: shared memory (sm), loopback (self), and available network devices
- UCX typically tries transports in order: `rc` (InfiniBand RC), `ud` (InfiniBand UD), `dc` (InfiniBand DC), `tcp`, `sm`, `self`
- This is often the best starting point - let the library auto-configure

**To see what's actually being loaded at runtime:**
```bash
# See UCC TL initialization
export UCC_LOG_LEVEL=INFO
mpirun -np 4 ./time-ucc 2>&1 | grep -i "TL\|transport"

# See UCX transport selection
export UCX_LOG_LEVEL=INFO
mpirun -np 4 ./time-ucc 2>&1 | grep -i "selected\|transport\|device"

# See both (verbose)
export UCC_LOG_LEVEL=INFO UCX_LOG_LEVEL=INFO
mpirun -np 4 ./time-ucc 2>&1 | tee output.log
```

### Understanding Transport Layers

**UCC Transport Layers (TL):**
- `ucp` - Uses UCX for communication (most common)
- `nccl` - NVIDIA NCCL library (GPU collectives)
- `sharp` - Mellanox SHARP (in-network computing)
- `self` - Single process (testing/fallback)
- `shm` - Shared memory between processes

**UCX Transports:**
- `rc` - InfiniBand Reliable Connection (best latency)
- `ud` - InfiniBand Unreliable Datagram (good for small messages)
- `dc` - InfiniBand Dynamically Connected (scalable)
- `tcp` - TCP/IP sockets (fallback, widely available)
- `sm` - POSIX shared memory (intra-node)
- `self` - Loopback (intra-process)
- `cuda` - NVIDIA CUDA IPC (GPU-to-GPU)
- `rocm` - AMD ROCm (GPU-to-GPU)

## Configuration

Edit `time-ucc.c` to modify:

```c
#define NR_ITER     10    // Number of iterations
#define SKIP        1     // Warmup iterations to skip
```

## Performance Tips

1. **Warmup iterations**: Use SKIP to exclude cold-start overhead
2. **Run multiple times**: For consistent results, run the benchmark multiple times
3. **Transport selection**: Experiment with different UCX_TLS configurations
4. **Network isolation**: For accurate measurements, run on isolated network
5. **Process placement**: Use proper MPI process binding for consistency

## Understanding Results

- **High variance**: May indicate system noise or inconsistent network performance
- **First iteration slower**: This is normal - warmup iterations help
- **Context vs Team time**: Context creation typically slower than team creation
- **Scaling**: Time should remain relatively constant with process count (for good scalability)

## Troubleshooting

### UCC not found
```bash
export LD_LIBRARY_PATH=/path/to/ucc/lib:$LD_LIBRARY_PATH
export CPATH=/path/to/ucc/include:$CPATH
```

### Transport issues
Check available transports:
```bash
ucx_info -d
```

Enable verbose output:
```bash
export UCC_LOG_LEVEL=INFO
export UCX_LOG_LEVEL=INFO
```

## Key Improvements from Original Code

1. ✅ **Dynamic TL detection**: Uses `dl_iterate_phdr()` to detect **actually loaded** transport layers at runtime
2. ✅ **Statistical analysis**: Added variance, stddev, min, max
3. ✅ **UCC/UCX visibility**: Shows both UCC TLs (`libucc_tl_*.so`) and UCX transports (`libuct_*.so`)
4. ✅ **Better output formatting**: Structured, readable output with transport descriptions
5. ✅ **Memory cleanup**: Proper resource deallocation
6. ✅ **Pure MPI**: Removed SHMEM dependencies
7. ✅ **Per-iteration display**: See timing for each iteration
8. ✅ **Better error handling**: More informative error messages
9. ✅ **Transport probe script**: Helper script to detect available TLs on your system

## Technical Details

### Dynamic Transport Detection via `dl_iterate_phdr()`

The benchmark uses a novel approach to detect actually loaded transport layers by examining the process's dynamic linker state:

```c
#include <link.h>

// Callback to find UCC TL libraries
static int find_ucc_tls_callback(struct dl_phdr_info *info, size_t size, void *data) {
    const char *name = info->dlpi_name;
    // Look for libucc_tl_*.so pattern
    if (strncmp(basename, "libucc_tl_", 10) == 0) {
        // Extract TL name and store it
    }
    return 0;
}

// Iterate through all loaded shared libraries
dl_iterate_phdr(find_ucc_tls_callback, &tl_list);
```

**How It Works:**
1. `dl_iterate_phdr()` walks through all shared libraries loaded in the process
2. Searches for libraries matching `libucc_tl_*.so` (UCC transport layers)
3. Searches for libraries matching `libuct_*.so` (UCX transports)
4. Extracts the transport name from the library filename
5. Displays the actual loaded transports with descriptions

**Why This Approach?**
- ✅ Shows **actual runtime state**, not just configuration
- ✅ No dependency on log parsing or environment variables
- ✅ Works regardless of log level settings
- ✅ Provides ground truth about what's loaded in memory
- ✅ Can detect transports even if not explicitly configured

### UCC API Usage for Context Information

The benchmark also uses UCC API calls for additional context information:

```c
// Query library attributes
ucc_lib_attr_t lib_attr;
lib_attr.mask = UCC_LIB_ATTR_FIELD_THREAD_MODE;
ucc_lib_get_attr(lib, &lib_attr);

// Query context attributes
ucc_context_attr_t ctx_attr;
ctx_attr.mask = UCC_CONTEXT_ATTR_FIELD_CTX_ADDR | 
                UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN |
                UCC_CONTEXT_ATTR_FIELD_WORK_BUFFER_SIZE;
ucc_context_get_attr(ctx, &ctx_attr);
```

**Available Context Attributes:**
- `UCC_CONTEXT_ATTR_FIELD_CTX_ADDR` - Context address for communication
- `UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN` - Length of context address
- `UCC_CONTEXT_ATTR_FIELD_WORK_BUFFER_SIZE` - Size of work buffers used

### Library Naming Conventions

**UCC Transport Layers:**
- Pattern: `libucc_tl_<name>.so`
- Examples: `libucc_tl_ucp.so`, `libucc_tl_nccl.so`, `libucc_tl_sharp.so`

**UCX Transports:**
- Pattern: `libuct_<name>.so`
- Examples: `libuct_ib.so`, `libuct_tcp.so`, `libuct_sm.so`, `libuct_cuda.so`

## License

See project license.

