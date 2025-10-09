# Implementation Notes: Dynamic Transport Layer Detection

## Overview

This benchmark uses `dl_iterate_phdr()` to dynamically detect which UCC and UCX transport layers are actually loaded at runtime, providing ground truth about the transport configuration without relying on environment variables or log parsing.

## How It Works

### 1. Dynamic Linker Introspection

```c
#include <link.h>  // For dl_iterate_phdr()

// dl_iterate_phdr() walks through all loaded shared libraries
// and calls our callback for each one
dl_iterate_phdr(find_ucc_tls_callback, &tl_list);
```

### 2. Pattern Matching

The callback searches for specific library naming patterns:

**UCC Transport Layers:**
```
Pattern: libucc_tl_<name>.so
Examples:
  - libucc_tl_ucp.so      → "ucp" 
  - libucc_tl_nccl.so     → "nccl"
  - libucc_tl_sharp.so    → "sharp"
```

**UCX Transports:**
```
Pattern: libuct_<name>.so
Examples:
  - libuct_ib.so          → "ib"
  - libuct_tcp.so         → "tcp"
  - libuct_sm.so          → "sm"
  - libuct_cuda.so        → "cuda"
```

### 3. Extraction Algorithm

```c
static int find_ucc_tls_callback(struct dl_phdr_info *info, size_t size, void *data) {
    const char *name = info->dlpi_name;
    
    // Get basename (filename without path)
    const char *basename = strrchr(name, '/');
    basename = basename ? basename + 1 : name;
    
    // Check for UCC TL pattern
    if (strncmp(basename, "libucc_tl_", 10) == 0) {
        // Extract name between "libucc_tl_" and ".so"
        const char *tl_start = basename + 10;
        const char *tl_end = strstr(tl_start, ".so");
        
        // Store the TL name
        strncpy(tl_list->tl_names[count], tl_start, len);
    }
    return 0;
}
```

## Timing Considerations

### When Are TLs Loaded?

**UCC Transport Layers** are loaded when:
1. `ucc_init()` is called (library initialization)
2. `ucc_context_create()` is called (context creation)
3. The specific TL is needed for an operation

**Important:** Our benchmark calls `dl_iterate_phdr()` **after** context creation, ensuring all TLs are already loaded and detectable.

### Code Flow

```
MPI_Init()
  ↓
ucc_init()                    // UCC library initialization
  ↓
ucc_context_create()          // TLs get loaded HERE
  ↓
print_ucc_info()              // We detect TLs here
  ↓
dl_iterate_phdr()             // Walks loaded libraries
```

## Advantages Over Other Methods

### vs. Environment Variable Checking
```c
char *tls = getenv("UCC_TLS");
// ❌ Only shows what was requested
// ❌ Doesn't show what actually loaded
// ❌ May be NULL even if TLs are loaded
```

### vs. Log Parsing
```bash
export UCC_LOG_LEVEL=INFO
# ❌ Requires specific log level
# ❌ Brittle (log format can change)
# ❌ Output mixed with other logs
# ❌ Requires post-processing
```

### vs. dl_iterate_phdr()
```c
dl_iterate_phdr(callback, &data);
// ✅ Shows actual runtime state
// ✅ No configuration needed
// ✅ Works regardless of log levels
// ✅ Programmatic access to data
// ✅ Ground truth about loaded libraries
```

## Example Output

```
Loaded Transport Layers (via dl_iterate_phdr):
  Found 2 loaded UCC TLs:
    - ucp (UCX Protocol - general purpose)
    - nccl (NVIDIA NCCL - GPU collectives)
  Requested via UCC_TLS: ucp,nccl

Loaded UCX Transports (via dl_iterate_phdr):
  Found 3 loaded UCX transports:
    - ib (InfiniBand/Mellanox)
    - sm (Shared Memory)
    - self (Loopback)
  Requested via UCX_TLS: rc,self,sm
```

## Comparison: Requested vs Loaded

The output shows both what was **requested** (env vars) and what's **actually loaded**:

```
Requested via UCC_TLS: ucp,nccl,sharp
Actually loaded: ucp, nccl

Reason: SHARP library not available on this system
```

This is incredibly useful for debugging configuration issues!

## Platform Compatibility

**Linux:** ✅ Full support (`link.h` is part of glibc)

**Other Platforms:**
- MacOS: Has `dyld` equivalent but different API
- Windows: Has `EnumProcessModules()` but different API
- Solaris: Has similar functionality

Current implementation is Linux-specific. For portability, could add:
```c
#ifdef __linux__
  dl_iterate_phdr(callback, data);
#elif __APPLE__
  // Use dyld APIs
#elif _WIN32
  // Use EnumProcessModules
#endif
```

## Limitations

1. **Lazy Loading**: Some TLs might be loaded lazily on first use. Our benchmark loads them during context creation, so this shouldn't be an issue.

2. **Versioned Libraries**: Libraries like `libucc_tl_ucp.so.1.0` will still be detected (we only check up to `.so`).

3. **Static Linking**: This approach only works with dynamic libraries. If UCC were statically linked, we'd need a different approach.

## Future Enhancements

1. **Detection of CL (Collective Layers)**:
   ```c
   // Could also detect libucc_cl_*.so
   if (strncmp(basename, "libucc_cl_", 10) == 0) {
       // Collective layers: basic, hier, etc.
   }
   ```

2. **Version Information**:
   ```c
   // Extract version from libucc_tl_ucp.so.1.15.0
   parse_version(basename);
   ```

3. **Load Order**:
   ```c
   // Track order in which libraries were loaded
   tl_list->load_order[i] = load_sequence++;
   ```

## References

- `man dl_iterate_phdr` - Linux documentation
- UCC source: `src/components/tl/` - TL implementations
- UCX source: `src/uct/` - UCT transport implementations

## Credits

This approach was suggested to improve upon environment variable checking and log parsing methods, providing a more reliable and programmatic way to determine the actual transport layer configuration.

