CC=mpicc
CFLAGS=-O0 -g3 -I/work/nvidia/install.ucc/include
LDFLAGS=-L/work/nvidia/install.ucc/lib -lucc

.PHONY: all clean

all: mpi_bench_a2a

mpi_bench_a2a: mpi_bench_a2a.c 

clean:
	rm mpi_bench_a2a
