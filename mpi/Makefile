CC=mpicc
CFLAGS=-O0 -g3 -I/work/nvidia/install.ucc/include
LDFLAGS=-L/work/nvidia/install.ucc/lib -lucc

.PHONY: all clean

all: mpi_bench_a2a ucc_bench_a2a

ucc_bench_a2a: mpi_bench_a2a.c
	${CC} ${CFLAGS} -DWITH_UCC mpi_bench_a2a.c -o ucc_bench_a2a ${LDFLAGS}

mpi_bench_a2a: mpi_bench_a2a.c 

clean:
	rm mpi_bench_a2a ucc_bench_a2a
