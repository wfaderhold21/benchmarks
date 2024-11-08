CC=shmemcc
UCC_DIR=${HOME}/build-a2av/install.ucc
CFLAGS=-O3 -g -I${UCC_DIR}/include
LDFLAGS=-L${UCC_DIR}/lib -lucp -lucs -lucc

.PHONY: clean

bw-pipeline: bw-pipeline.c
	${CC} ${CFLAGS} bw-pipeline.c -o $@ ${LDFLAGS}

clean:
	rm bw-pipeline
