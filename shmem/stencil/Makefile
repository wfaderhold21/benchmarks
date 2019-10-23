NVCC=nvcc
CC=oshcc
PREFIX=/home/wa9/work/test-ompi-build/install.ompi-sharp
CFLAGS=-O2 -g -I${PREFIX}/include
LDFLAGS=-L${PREFIX}/lib -loshmem  

.PHONY: all clean

all: 5pt

5pt:    5pt-2d-shmem.c common.h
	${CC} ${CFLAGS} 5pt-2d-shmem.c -o 5pt ${LDFLAGS}

clean:
	rm 5pt
