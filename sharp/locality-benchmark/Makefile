SHARP_PREFIX=/home/ferrol/install.sharp
UCX_PREFIX=${SHARP_PREFIX}
RTE_PREFIX=${SHARP_PREFIX}

CFLAGS = -O3 -I${SHARP_PREFIX}/include -I${SHARP_PREFIX}/include/sharp -I${SHARP_PREFIX}/include/sharp/utils -I${UCX_PREFIX}/include -I${RTE_PREFIX}/include
CC = mpicc
LDFLAGS = -L${SHARP_PREFIX}/lib -L${UCX_PREFIX}/lib -L${RTE_PREFIX}/lib -lsharp -lpmem -lvmem -lmemkind -loshmem 

.PHONY: clean

all: locality

locality: locality.c 
	${CC} ${CFLAGS} -DLAPTOP locality.c -o locality ${LDFLAGS}

clean:
	rm locality
