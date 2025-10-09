#!/bin/bash
# Script to probe available UCC and UCX transports

echo "========================================"
echo "UCC/UCX Transport Layer Probe"
echo "========================================"
echo ""

# Check if UCX is installed
if command -v ucx_info &> /dev/null; then
    echo "=== UCX Installation Found ==="
    ucx_info -v 2>&1 | head -3
    echo ""
    
    echo "=== Available UCX Transports ==="
    ucx_info -d 2>&1 | grep -E "Transport:|Device:" | head -20
    echo ""
    
    echo "=== UCX Configuration Options ==="
    echo "Available UCX_TLS options:"
    ucx_info -c 2>&1 | grep "UCX_TLS" | head -5
    echo ""
else
    echo "UCX tools not found. Please install UCX or add to PATH."
    echo ""
fi

# Check if UCC tools are available
if command -v ucc_info &> /dev/null; then
    echo "=== UCC Installation Found ==="
    ucc_info -v 2>&1 | head -3
    echo ""
    
    echo "=== Available UCC Transport Layers ==="
    ucc_info -c 2>&1 | grep "UCC_TLS" | head -5
    echo ""
else
    echo "UCC tools not found (ucc_info). This is normal if not installed separately."
    echo ""
fi

# Check environment
echo "=== Current Environment Variables ==="
env | grep -E "^UCC_|^UCX_|^NCCL_" | sort
if [ $? -ne 0 ]; then
    echo "(None set - will use defaults)"
fi
echo ""

# Check for InfiniBand
echo "=== InfiniBand Devices ==="
if command -v ibstat &> /dev/null; then
    ibstat -l 2>/dev/null || echo "No InfiniBand devices found"
elif [ -d /sys/class/infiniband ]; then
    ls /sys/class/infiniband/ 2>/dev/null || echo "No InfiniBand devices found"
else
    echo "InfiniBand not detected"
fi
echo ""

# Check for NVIDIA GPUs
echo "=== NVIDIA GPU Devices ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi -L 2>/dev/null || echo "No NVIDIA GPUs found"
else
    echo "NVIDIA GPUs not detected or nvidia-smi not in PATH"
fi
echo ""

# Check for AMD GPUs
echo "=== AMD GPU Devices ==="
if command -v rocm-smi &> /dev/null; then
    rocm-smi -i 2>/dev/null || echo "No AMD GPUs found"
else
    echo "AMD ROCm not detected"
fi
echo ""

echo "========================================"
echo "Probe complete!"
echo "========================================"
echo ""
echo "Recommended configurations based on detected hardware:"
echo ""

# Provide recommendations
if ibstat -l &> /dev/null || [ -d /sys/class/infiniband ]; then
    echo "InfiniBand detected - Recommended UCX_TLS:"
    echo "  export UCX_TLS=rc,sm,self"
    echo ""
fi

if nvidia-smi -L &> /dev/null; then
    echo "NVIDIA GPUs detected - Consider UCC NCCL TL:"
    echo "  export UCC_TLS=nccl,ucp"
    echo "  export UCX_TLS=rc,cuda_copy,cuda_ipc,sm,self"
    echo ""
fi

echo "For CPU-only with InfiniBand:"
echo "  export UCC_TLS=ucp"
echo "  export UCX_TLS=rc,sm,self"
echo ""

echo "For TCP/IP networks:"
echo "  export UCC_TLS=ucp"
echo "  export UCX_TLS=tcp,sm,self"
echo ""

echo "For single-node testing:"
echo "  export UCC_TLS=ucp"
echo "  export UCX_TLS=sm,self"

