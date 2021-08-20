CC=mpicc
UCX_PREFIX=/work/nvidia/install.ucc
PMIX_PREFIX=/work/nvidia/install.pmix
CFLAGS=-g -O0 -DUSE_MPI=1 -I${UCX_PREFIX}/include -I${PMIX_PREFIX}/include  
LDFLAGS=-L${UCX_PREFIX}/lib -lucp -luct -lucs -lucm -lpmix


.PHONY: all clean
all: wireup

wireup: wireup.c

clean:
	rm wireup
