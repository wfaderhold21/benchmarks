CC=oshcc
CFLAGS=-O2 -g -Wall -fopenmp
LDFLAGS=-lmpi 

.PHONY:	all clean

all: alltoall alltoall-hint alltoall-mt alltoall_debug

alltoall:	alltoall.c
	$(CC) $(CFLAGS) -o alltoall alltoall.c $(LDFLAGS)

alltoall-hint:	alltoall.c
	$(CC) $(CFLAGS) -DWITH_HINTS -o alltoall-hint alltoall.c $(LDFLAGS)

alltoall-mt:	alltoall-mt.c
	$(CC) $(CFLAGS) -DWITH_HINTS -o alltoall-mt alltoall-mt.c $(LDFLAGS)

alltoall_debug:	alltoall.c
	$(CC) $(CFLAGS) -DALLTOALL_DEBUG -o alltoall_debug alltoall.c $(LDFLAGS)

clean:
	rm alltoall alltoall-hint alltoall_debug
