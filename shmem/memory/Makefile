CC=oshcc
CFLAGS=-O2 -g -Wall
LDFLAGS=-lmpi

.PHONY:	all clean

all: shmem_memory_usage

shmem_memory_usage: shmem_memory_usage.c
	$(CC) $(CFLAGS) -o shmem_memory_usage shmem_memory_usage.c $(LDFLAGS)

clean:
	rm -f shmem_memory_usage
