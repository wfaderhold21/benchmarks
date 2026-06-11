CC=mpicc
CFLAGS=-O0 -g3 -I/work/nvidia/install.ucc/include
LDFLAGS=-L/work/nvidia/install.ucc/lib -lucc
RDMA_LDFLAGS=-libverbs
GCC=gcc

.PHONY: all clean

RDMA_AVAILABLE := $(shell test -f /usr/include/infiniband/verbs.h && echo 1)
RDMA_TARGETS := $(if $(RDMA_AVAILABLE),rdma_alltoall_bench rdma_alltoall_simple,)

all: mpi_bench_a2a ucc_bench_a2a rdma_alltoall_demo $(RDMA_TARGETS)

ucc_bench_a2a: mpi_bench_a2a.c
	${CC} ${CFLAGS} -DWITH_UCC mpi_bench_a2a.c -o ucc_bench_a2a ${LDFLAGS}

mpi_bench_a2a: mpi_bench_a2a.c
	${CC} ${CFLAGS} mpi_bench_a2a.c -o mpi_bench_a2a ${LDFLAGS}

rdma_alltoall_bench: rdma_alltoall_bench.c
	${CC} ${CFLAGS} rdma_alltoall_bench.c -o rdma_alltoall_bench ${RDMA_LDFLAGS}

rdma_alltoall_simple: rdma_alltoall_simple.c
	${GCC} ${CFLAGS} rdma_alltoall_simple.c -o rdma_alltoall_simple ${RDMA_LDFLAGS}

rdma_alltoall_demo: rdma_alltoall_demo.c
	${GCC} ${CFLAGS} rdma_alltoall_demo.c -o rdma_alltoall_demo

clean:
	rm -f mpi_bench_a2a ucc_bench_a2a rdma_alltoall_bench rdma_alltoall_simple rdma_alltoall_demo
