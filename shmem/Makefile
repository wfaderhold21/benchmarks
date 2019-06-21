CC=oshcc
CFLAGS=-O2 
LDFLAGS=-lmpi

.PHONY:	all clean

all: alltoall alltoall_debug

alltoall:	alltoall.c
	$(CC) $(CFLAGS) -o alltoall alltoall.c $(LDFLAGS)

alltoall_debug:	alltoall.c
	$(CC) $(CFLAGS) -DALLTOALL_DEBUG -o alltoall_debug alltoall.c $(LDFLAGS)

clean:
	rm alltoall alltoall_debug
