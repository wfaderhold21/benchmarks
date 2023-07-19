CC=shmemcc
CFLAGS=-O2 -g
LDFLAGS=-lucc

.PHONY: all clean

all: ucc_bench_a2av ucc_bench_a2av_basic

ucc_bench_a2av: ucc_bench_a2av.c 

ucc_bench_a2av_basic: ucc_bench_a2av.c 
	${CC} ${CFLAGS} -DWITH_BASIC ucc_bench_a2av.c -o ucc_bench_a2av_basic ${LDFLAGS}

clean:
	rm ucc_bench_a2av ucc_bench_a2av_basic
