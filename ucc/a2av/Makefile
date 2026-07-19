CC=shmemcc
CFLAGS=-O2 -g -I../../common
LDFLAGS=-lucc
TRANSPORT_OBJS=../../common/transport_detect.o

.PHONY: all clean

all: ucc_bench_a2av ucc_bench_a2av_basic ucc_bench_a2av_verify ucc_bench_a2av_mpi

../../common/transport_detect.o: ../../common/transport_detect.c ../../common/transport_detect.h
	$(CC) $(CFLAGS) -c $< -o $@

ucc_bench_a2av: ucc_bench_a2av.c ../../common/transport_detect.o
	${CC} ${CFLAGS} ucc_bench_a2av.c ../../common/transport_detect.o -o ucc_bench_a2av ${LDFLAGS}

ucc_bench_a2av_basic: ucc_bench_a2av.c ../../common/transport_detect.o
	${CC} ${CFLAGS} -DWITH_BASIC ucc_bench_a2av.c ../../common/transport_detect.o -o ucc_bench_a2av_basic ${LDFLAGS}

ucc_bench_a2av_verify: ucc_bench_a2av.c ../../common/transport_detect.o
	${CC} ${CFLAGS} -DWITH_VERIFY ucc_bench_a2av.c ../../common/transport_detect.o -o ucc_bench_a2av_verify ${LDFLAGS}

ucc_bench_a2av_mpi: ucc_bench_a2av_mpi.c ../../common/transport_detect.o
	${CC} ${CFLAGS} ucc_bench_a2av_mpi.c ../../common/transport_detect.o -o ucc_bench_a2av_mpi ${LDFLAGS}

clean:
	rm -f ucc_bench_a2av ucc_bench_a2av_basic ucc_bench_a2av_verify ucc_bench_a2av_mpi ../../common/transport_detect.o
