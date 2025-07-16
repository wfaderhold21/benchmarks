CC=shmemcc
CFLAGS=-O2 -g
LDFLAGS=-lucc

.PHONY: all clean

all: ucc_bench_a2a ucc_bench_a2a_basic ucc_bench_a2a_verify

ucc_bench_a2a: ucc_bench_a2a.c

ucc_bench_a2a_basic: ucc_bench_a2a.c
	${CC} ${CFLAGS} -DWITH_BASIC ucc_bench_a2a.c -o ucc_bench_a2a_basic ${LDFLAGS}

ucc_bench_a2a_verify: ucc_bench_a2a.c
	${CC} ${CFLAGS} -DWITH_VERIFY ucc_bench_a2a.c -o ucc_bench_a2a_verify ${LDFLAGS}

clean:
	rm ucc_bench_a2a ucc_bench_a2a_basic ucc_bench_a2a_verify
