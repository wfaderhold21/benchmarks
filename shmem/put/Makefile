CC=oshcc
CFLAGS=-O2 -g -Wall
LDFLAGS=-lmpi

.PHONY:	all clean

all: put put-hint

put:	put.c
	$(CC) $(CFLAGS) -o put put.c $(LDFLAGS)

put-hint:	put.c
	$(CC) $(CFLAGS) -DWITH_HINTS -o put-hint put.c $(LDFLAGS)

clean:
	rm put put-hint
