CC=mpicc
UCX_PREFIX=${HOME}/install.ucx
CFLAGS=-g3 -O3 -DUSE_MPI -I${UCX_PREFIX}/include 
LDFLAGS=-L${UCX_PREFIX}/lib -lucp -luct -lucs -lucm


.PHONY: all clean
all: wireup

wireup: wireup.c mpi.c
	${CC} ${CFLAGS} wireup.c mpi.c -o wireup ${LDFLAGS}

clean:
	rm wireup
