SHARP_PREFIX=/ccs/proj/csc221/ferrol/rhea/install.sharp
UCX_PREFIX=/ccs/proj/csc221/ferrol/rhea/install.ucx
RTE_PREFIX=/ccs/proj/csc221/ferrol/rhea/install.rte

CFLAGS = -O3 -DRHEA -I${SHARP_PREFIX}/include -I${SHARP_PREFIX}/include/sharp -I${SHARP_PREFIX}/include/sharp/utils -I${UCX_PREFIX}/include -I${RTE_PREFIX}/include
CC = mpicc
LDFLAGS = -L${SHARP_PREFIX}/lib -L${UCX_PREFIX}/lib -L${RTE_PREFIX}/lib -lsharp -lpmem -lvmem -lmemkind -lucs -luct -lucm -lucp -lrte

.PHONY: clean

all: locality-rhea

locality-rhea: locality.c 
	${CC} ${CFLAGS} locality.c -o locality-rhea ${LDFLAGS}

clean:
	rm locality-rhea
