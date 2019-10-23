CC=oshcc
CFLAGS=-O2 -g -Wall -fopenmp
LDFLAGS=-lmpi 

.PHONY:	all clean

all: alltoall alltoall-hint alltoall-mt-near alltoall-mt-far alltoall-mt-near-hint alltoall-mt-far-hint alltoall-mt-local-hint alltoall-mt-interleave-hint alltoall_debug

alltoall:	alltoall.c
	$(CC) $(CFLAGS) -o alltoall alltoall.c $(LDFLAGS)

alltoall-hint:	alltoall.c
	$(CC) $(CFLAGS) -DWITH_HINTS -o alltoall-hint alltoall.c $(LDFLAGS)

alltoall-mt-near:	alltoall-mt.c
	$(CC) $(CFLAGS) -DNEAR -o alltoall-mt-near alltoall-mt.c $(LDFLAGS)

alltoall-mt-far:	alltoall-mt.c
	$(CC) $(CFLAGS) -o alltoall-mt-far alltoall-mt.c $(LDFLAGS)


alltoall-mt-near-hint:	alltoall-mt.c
	$(CC) $(CFLAGS) -DNEAR=1 -DWITH_HINTS -o alltoall-mt-near-hint alltoall-mt.c $(LDFLAGS)

alltoall-mt-far-hint:	alltoall-mt.c
	$(CC) $(CFLAGS) -DFAR=1 -DWITH_HINTS -o alltoall-mt-far-hint alltoall-mt.c $(LDFLAGS)

alltoall-mt-local-hint:	alltoall-mt.c
	$(CC) $(CFLAGS) -DLOCAL=1 -DWITH_HINTS -o alltoall-mt-local-hint alltoall-mt.c $(LDFLAGS)

alltoall-mt-interleave-hint:	alltoall-mt.c
	$(CC) $(CFLAGS) -DINTERLEAVE=1 -DWITH_HINTS -o alltoall-mt-interleave-hint alltoall-mt.c $(LDFLAGS)


alltoall_debug:	alltoall.c
	$(CC) $(CFLAGS) -DALLTOALL_DEBUG -o alltoall_debug alltoall.c $(LDFLAGS)

clean:
	rm alltoall alltoall-hint alltoall_debug alltoall-mt-near alltoall-mt-far alltoall-mt-near-hint alltoall-mt-far-hint alltoall-mt-local-hint alltoall-mt-interleave-hint
