CC=shmemcc
CFLAGS=-O3 -g
LDFLAGS=-lucc

.PHONY: all clean

all: ucc_bench_a2a ucc_bench_a2a_basic ucc_bench_a2a_verify ucc_bench_a2a_mpi ucc_bench_a2a_mpi_ar ucc_bench_a2a_shrink

ucc_bench_a2a: ucc_bench_a2a.c

ucc_bench_a2a_basic: ucc_bench_a2a.c
	${CC} ${CFLAGS} -DWITH_BASIC ucc_bench_a2a.c -o ucc_bench_a2a_basic ${LDFLAGS}

ucc_bench_a2a_verify: ucc_bench_a2a.c
	${CC} ${CFLAGS} -DWITH_VERIFY ucc_bench_a2a.c -o ucc_bench_a2a_verify ${LDFLAGS}

ucc_bench_a2a_mpi: ucc_bench_a2a_mpi.c
	${CC} ${CFLAGS} ucc_bench_a2a_mpi.c -o ucc_bench_a2a_mpi ${LDFLAGS}

ucc_bench_a2a_mpi_ar: ucc_bench_a2a_mpi_ar.c
	${CC} ${CFLAGS} ucc_bench_a2a_mpi_ar.c -o ucc_bench_a2a_mpi_ar ${LDFLAGS}

UCC_SRC ?= $(HOME)/ucc-resilience

ucc_bench_a2a_shrink: ucc_bench_a2a_shrink.c
	${CC} ${CFLAGS} -I$(UCC_SRC)/src ucc_bench_a2a_shrink.c -o ucc_bench_a2a_shrink ${LDFLAGS}

clean:
	rm ucc_bench_a2a ucc_bench_a2a_basic ucc_bench_a2a_verify ucc_bench_a2a_mpi ucc_bench_a2a_mpi_ar ucc_bench_a2a_shrink
