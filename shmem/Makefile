CC=oshcc
CFLAGS=-O2 -g -Wall
LDFLAGS=-lmpi

.PHONY:	all clean

all: alltoall alltoall-hint alltoall_debug put put-hint

alltoall:	alltoall.c
	$(CC) $(CFLAGS) -o alltoall alltoall.c $(LDFLAGS)

alltoall-hint:	alltoall.c
	$(CC) $(CFLAGS) -DWITH_HINTS -o alltoall-hint alltoall.c $(LDFLAGS)

alltoall_debug:	alltoall.c
	$(CC) $(CFLAGS) -DALLTOALL_DEBUG -o alltoall_debug alltoall.c $(LDFLAGS)

put:	put.c
	$(CC) $(CFLAGS) -o put put.c $(LDFLAGS)

put-hint:	put.c
	$(CC) $(CFLAGS) -DWITH_HINTS -o put-hint put.c $(LDFLAGS)

clean:
	rm alltoall alltoall-hint alltoall_debug put put-hint
