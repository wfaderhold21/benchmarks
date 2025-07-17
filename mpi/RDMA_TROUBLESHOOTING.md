# RDMA Troubleshooting Guide

## Common RDMA Errors and Solutions

### IBV_WC_WR_FLUSH_ERR (Status 5) - Work Request Flushed Error

**Error Description:**
```
Work completion failed with status 5
```

**Root Cause:**
The `IBV_WC_WR_FLUSH_ERR` occurs when a work request is flushed due to:
1. **Queue Pair (QP) State Transitions**: QP is not in the correct state for posting work requests
2. **Multiple Connections on Single QP**: Trying to use one QP for multiple remote connections
3. **QP Connection Issues**: QP not properly connected to remote QP
4. **Memory Registration Problems**: Memory region not properly registered or accessible

**Solution Implemented:**

#### 1. Multiple QPs Architecture
**Problem**: Using a single QP for all connections
```c
// WRONG: Single QP for all connections
struct rdma_context {
    struct ibv_qp *qp;  // Only one QP
};
```

**Solution**: Create separate QP for each connection
```c
// CORRECT: Multiple QPs for multiple connections
struct rdma_context {
    struct ibv_qp **qps;  // Array of QPs
    int num_qps;
};
```

#### 2. Proper QP Assignment
Each rank creates `(num_ranks - 1)` QPs and assigns them to specific connections:

```c
// QP assignment logic
if (i < my_rank) {
    connections[i].qp_index = i;
    connections[i].remote_qp_num = remote_info[i].qp_nums[my_rank - 1];
} else if (i > my_rank) {
    connections[i].qp_index = i - 1;
    connections[i].remote_qp_num = remote_info[i].qp_nums[my_rank];
} else {
    connections[i].qp_index = -1;  // No QP needed for self
}
```

#### 3. Enhanced Error Reporting
Added detailed error messages to identify which QP is causing issues:

```c
if (wc.status != IBV_WC_SUCCESS) {
    fprintf(stderr, "Work completion failed with status %d (QP %d)\n", 
            wc.status, qp_idx);
    return -1;
}
```

### Other Common RDMA Errors

#### IBV_WC_RNR_RETRY_EXC_ERR (Status 23)
**Cause**: Remote Not Ready - receiver queue is full
**Solution**: Increase receive queue size or add flow control

#### IBV_WC_RETRY_EXC_ERR (Status 22)
**Cause**: Too many retries due to network congestion
**Solution**: Increase retry count or reduce network load

#### IBV_WC_BAD_RESP_ERR (Status 13)
**Cause**: Invalid response from remote QP
**Solution**: Check QP state and connection parameters

#### IBV_WC_LOC_QP_OP_ERR (Status 4)
**Cause**: Local QP operation error
**Solution**: Check QP state transitions and work request parameters

### Debugging Steps

#### 1. Check QP States
```c
struct ibv_qp_attr attr;
struct ibv_qp_init_attr init_attr;
if (ibv_query_qp(qp, &attr, IBV_QP_STATE, &init_attr) == 0) {
    printf("QP state: %d\n", attr.qp_state);
}
```

#### 2. Verify Memory Registration
```c
// Check if memory region is valid
if (rdma_ctx->mr && rdma_ctx->mr->lkey && rdma_ctx->mr->rkey) {
    printf("Memory region valid: lkey=%u, rkey=%u\n", 
           rdma_ctx->mr->lkey, rdma_ctx->mr->rkey);
}
```

#### 3. Validate Connection Parameters
```c
// Check remote QP number and memory keys
for (int i = 0; i < num_ranks; i++) {
    if (i != my_rank) {
        printf("Rank %d: remote_qp=%u, remote_rkey=%u\n", 
               i, connections[i].remote_qp_num, connections[i].remote_rkey);
    }
}
```

### Best Practices

#### 1. QP Management
- Create separate QPs for each connection
- Ensure proper QP state transitions (INIT → RTR → RTS)
- Use unique PSN (Packet Sequence Numbers) for each QP

#### 2. Memory Management
- Register memory with appropriate access flags
- Exchange memory keys (rkeys) between ranks
- Validate memory addresses before RDMA operations

#### 3. Error Handling
- Always check work completion status
- Implement retry mechanisms for transient errors
- Provide detailed error messages for debugging

#### 4. Synchronization
- Use MPI_Barrier for synchronization between iterations
- Ensure all ranks complete QP setup before starting operations
- Coordinate memory access to avoid conflicts

### Testing Recommendations

#### 1. Start with Small Messages
```c
#define MIN_MSG_SIZE 8  // Start with 8 bytes
```

#### 2. Use Fewer Ranks Initially
```c
mpirun -np 2 ./rdma_alltoall_bench  // Test with 2 ranks first
```

#### 3. Enable Verbose Logging
Add debug prints to track QP state transitions and memory operations.

#### 4. Validate Network Configuration
Ensure InfiniBand fabric is properly configured and accessible.

### Performance Considerations

#### 1. QP Creation Overhead
- Creating multiple QPs adds initialization overhead
- Consider QP pooling for frequently used connections

#### 2. Memory Registration
- Memory registration is expensive
- Reuse registered memory regions when possible

#### 3. Completion Queue Polling
- Poll completion queue efficiently
- Consider using completion events for better performance

### Example Fix Summary

The main fix for `IBV_WC_WR_FLUSH_ERR` involved:

1. **Architecture Change**: Single QP → Multiple QPs
2. **Connection Mapping**: Proper QP-to-connection assignment
3. **State Management**: Correct QP state transitions for each connection
4. **Error Reporting**: Enhanced debugging information

This ensures each RDMA operation uses a dedicated, properly connected QP, eliminating the flush errors caused by QP state conflicts. 