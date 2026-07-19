# benchmarks

Personal collection of HPC communication benchmarks covering SHMEM, MPI, UCX, UCC, and SHARP.

Actively maintained suites are marked with **active**; others are preserved for historical reference. This repository informs upstream discussions — see [openucx/ucc PRs](https://github.com/openucx/ucc/pulls) for related work.

## Layout

| Dir | What it measures | Build requirements |
|-----|------------------|--------------------|
| [`ucc/`](ucc/) **active** | UCC collectives: alltoall, alltoallv (fixed + variable), pipelined a2av bandwidth, context/team creation timing | MPI, UCC, UCX |
| [`ucx/`](ucx/) | UCX-level alltoall and full HPCAC exchange patterns | UCX, MPI |
| [`mpi/`](mpi/) | RDMA all-to-all variants, MPI alltoall benchmark, scalability analysis | MPI (RDMA-capable) |
| [`shmem/`](shmem/) | SHMEM put/stencil/nonblocking/cuda benchmarks; Graph500; comprehensive test suites | SHMEM (SYCL/MPI-based) |
| [`sharp/`](sharp/) | Mellanox SHARP locality benchmark for in-network reduction | SHARP, MPI |
| [`stream/`](stream/) | Endless copy/stream throughput micro-benchmark | none |
| [`archive/`](archive/) | Historical results (e.g. 2019 SHMEM alltoall experiments) | — |

## Quick start

```bash
# UCC alltoall
cd ucc/a2a && make && mpirun -np 4 ./ucc_bench_a2a

# UCC context/team timing
cd ucc/timing-ucc && make && mpirun -np 4 ./time-ucc

# UCX-level benchmarks
cd ucx/alltoall && make && mpirun -np 4 ./alltoall

# MPI RDMA alltoall
cd mpi && make && mpirun -np 4 ./rdma_alltoall_demo
```

## Output format

All UCC benchmarks emit CSV — see `docs/output-format.md`.

## Results

Historical results under [`archive/`](archive/).
