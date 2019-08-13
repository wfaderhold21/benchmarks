NVCC=nvcc
CC=g++
PREFIX=/home/cc/install.ompi-sharp
CFLAGS=-O2 -g -I${PREFIX}/include
LDFLAGS=-L/usr/local/cuda/lib64 -L${PREFIX}/lib -lcuda -loshmem -lcudart 

.PHONY: all clean

all: 5pt

5pt:    5pt-2d-cuda-shmem.c myfoo.o
	${CC} ${CFLAGS} 5pt-2d-cuda-shmem.c myfoo.o -o 5pt ${LDFLAGS}

myfoo.o:    myfoo.cu
	${NVCC} -c myfoo.cu 

clean:
	rm myfoo.o 5pt
