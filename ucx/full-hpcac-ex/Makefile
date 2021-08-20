CC=mpicc
UCX_PREFIX=${HOME}/install.ucc
PMIX_PREFIX=${HOME}/install.pmix
CFLAGS=-g3 -O3 -DUSE_MPI -I${UCX_PREFIX}/include -I${PMIX_PREFIX}/include  
LDFLAGS=-L${UCX_PREFIX}/lib -L${PMIX_PREFIX}/lib -lucp -luct -lucs -lucm -lpmix


.PHONY: all clean
all: wireup

wireup: wireup.c mpi.c pmix.c

clean:
	rm wireup
