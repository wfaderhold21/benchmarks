CC=shmemcc
HPCX=/work/nvidia/hpcx-v2.22-gcc-doca_ofed-ubuntu24.04-cuda12-x86_64
UCC_DIR=$(HPCX)/ucc
UCX_DIR=$(HPCX)/ucx
CFLAGS=-O3 -g -I$(UCC_DIR)/include
LDFLAGS=-L$(UCX_DIR)/lib -L$(UCC_DIR)/lib -lucp -lucs -lucc

.PHONY: clean

bw-pipeline: bw-pipeline.c
	${CC} ${CFLAGS} bw-pipeline.c -o $@ ${LDFLAGS}

clean:
	rm bw-pipeline
