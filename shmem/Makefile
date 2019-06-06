CC=oshcc
CFLAGS=-O2 
LDFLAGS=-lmpi

.PHONY:	all clean

all: alltoall

alltoall:	alltoall.c
	$(CC) $(CFLAGS) -o alltoall alltoall.c $(LDFLAGS)

clean:
	rm alltoall
