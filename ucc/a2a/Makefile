CC=shmemcc
CFLAGS=-O3 -g
LDFLAGS=-lucc

.PHONY: all clean

all: ucc_bench_a2a ucc_bench_a2a_basic ucc_bench_a2a_verify ucc_bench_a2a_mpi ucc_bench_a2a_mpi_ar ucc_bench_a2a_shrink
ifneq ($(CUDA),)
all: ucc_bench_a2a_cuda
endif

ucc_bench_a2a: ucc_bench_a2a.c

ucc_bench_a2a_basic: ucc_bench_a2a.c
	${CC} ${CFLAGS} -DWITH_BASIC ucc_bench_a2a.c -o ucc_bench_a2a_basic ${LDFLAGS}

ucc_bench_a2a_verify: ucc_bench_a2a.c
	${CC} ${CFLAGS} -DWITH_VERIFY ucc_bench_a2a.c -o ucc_bench_a2a_verify ${LDFLAGS}

ucc_bench_a2a_mpi: ucc_bench_a2a_mpi.c
	${CC} ${CFLAGS} ucc_bench_a2a_mpi.c -o ucc_bench_a2a_mpi ${LDFLAGS}

ucc_bench_a2a_mpi_ar: ucc_bench_a2a_mpi_ar.c
	${CC} ${CFLAGS} ucc_bench_a2a_mpi_ar.c -o ucc_bench_a2a_mpi_ar ${LDFLAGS}

ucc_bench_a2a_shrink: ucc_bench_a2a_shrink.c
	${CC} ${CFLAGS} ucc_bench_a2a_shrink.c -o ucc_bench_a2a_shrink ${LDFLAGS} || \
	    echo "WARNING: ucc_bench_a2a_shrink skipped (UCC FT APIs not available in this version)"

ifneq ($(CUDA),)
NVCC=nvcc
CUDA_INC=/usr/local/cuda/include
CUDA_LIB=/usr/local/cuda/lib64

ucc_bench_a2a_cuda: ucc_bench_a2a_cuda.c
	${NVCC} ${CFLAGS} -I${CUDA_INC} ucc_bench_a2a_cuda.c -o ucc_bench_a2a_cuda \
	    -L/usr/local/cuda/lib64 -lcudart ${LDFLAGS}
endif

clean:
	rm -f ucc_bench_a2a ucc_bench_a2a_basic ucc_bench_a2a_verify ucc_bench_a2a_mpi ucc_bench_a2a_mpi_ar ucc_bench_a2a_shrink ucc_bench_a2a_cuda
