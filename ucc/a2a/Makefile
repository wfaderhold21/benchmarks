CC=shmemcc
CFLAGS=-O3 -g -I../../common
LDFLAGS=-lucc -lm
COMMON_OBJS=../../common/transport_detect.o ../../common/nic_counters.o ../../common/qp_count.o ../../common/iio_pmu.o

.PHONY: all clean

all: ucc_bench_a2a ucc_bench_a2a_basic ucc_bench_a2a_verify ucc_bench_a2a_mpi ucc_bench_a2a_mpi_ar ucc_bench_a2a_shrink
ifeq ($(CUDA),1)
all: ucc_bench_a2a_cuda
endif

../../common/transport_detect.o: ../../common/transport_detect.c ../../common/transport_detect.h
	$(CC) $(CFLAGS) -c $< -o $@

../../common/nic_counters.o: ../../common/nic_counters.c ../../common/nic_counters.h ../../common/iio_pmu.h
	$(CC) $(CFLAGS) -c ../../common/nic_counters.c -o $@

../../common/qp_count.o: ../../common/qp_count.c ../../common/qp_count.h
	$(CC) $(CFLAGS) -c ../../common/qp_count.c -o $@

../../common/iio_pmu.o: ../../common/iio_pmu.c ../../common/iio_pmu.h
	$(CC) $(CFLAGS) -c ../../common/iio_pmu.c -o $@

ucc_bench_a2a: ucc_bench_a2a.c ../../common/transport_detect.o
	${CC} ${CFLAGS} ucc_bench_a2a.c ../../common/transport_detect.o -o $@ ${LDFLAGS}

ucc_bench_a2a_basic: ucc_bench_a2a.c ../../common/transport_detect.o
	${CC} ${CFLAGS} -DWITH_BASIC ucc_bench_a2a.c ../../common/transport_detect.o -o ucc_bench_a2a_basic ${LDFLAGS}

ucc_bench_a2a_verify: ucc_bench_a2a.c ../../common/transport_detect.o
	${CC} ${CFLAGS} -DWITH_VERIFY ucc_bench_a2a.c ../../common/transport_detect.o -o ucc_bench_a2a_verify ${LDFLAGS}

# MPI variant links the NIC/IIO/QP counter backends (node-wide ICMC measurement)
ucc_bench_a2a_mpi: ucc_bench_a2a_mpi.c $(COMMON_OBJS)
	${CC} ${CFLAGS} ucc_bench_a2a_mpi.c $(COMMON_OBJS) -o ucc_bench_a2a_mpi ${LDFLAGS}

ucc_bench_a2a_mpi_ar: ucc_bench_a2a_mpi_ar.c ../../common/transport_detect.o
	${CC} ${CFLAGS} ucc_bench_a2a_mpi_ar.c ../../common/transport_detect.o -o ucc_bench_a2a_mpi_ar ${LDFLAGS}

ucc_bench_a2a_shrink: ucc_bench_a2a_shrink.c ../../common/transport_detect.o
	${CC} ${CFLAGS} ucc_bench_a2a_shrink.c ../../common/transport_detect.o -o ucc_bench_a2a_shrink ${LDFLAGS} || \
	    echo "WARNING: ucc_bench_a2a_shrink skipped (UCC FT APIs not available in this version)"

ifeq ($(CUDA),1)
NVCC=nvcc
CUDA_INC=/usr/local/cuda/include
CUDA_LIB=/usr/local/cuda/lib64

ucc_bench_a2a_cuda: ucc_bench_a2a_cuda.c ../../common/transport_detect.o
	${NVCC} ${CFLAGS} -I${CUDA_INC} ucc_bench_a2a_cuda.c ../../common/transport_detect.o -o ucc_bench_a2a_cuda \
	    -L/usr/local/cuda/lib64 -lcudart ${LDFLAGS}
endif

clean:
	rm -f ucc_bench_a2a ucc_bench_a2a_basic ucc_bench_a2a_verify ucc_bench_a2a_mpi ucc_bench_a2a_mpi_ar ucc_bench_a2a_shrink ucc_bench_a2a_cuda ../../common/transport_detect.o ../../common/nic_counters.o ../../common/qp_count.o ../../common/iio_pmu.o
