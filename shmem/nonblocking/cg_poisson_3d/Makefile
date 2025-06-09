CC = oshcc
CFLAGS = -O3 -Wall
LDFLAGS = -lm

TARGET = cg_poisson_3d
SRC = cg_poisson_3d_oshmem.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean 