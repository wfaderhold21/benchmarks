# Makefile for UCC Context/Team Creation Benchmark

CC = mpicc
CFLAGS = -Wall -O2 -g -I/work/nvidia/install.ucc-test/include
LDFLAGS = -L/work/nvidia/install.ucc-test/lib -lucc -lucs -lopen-pal -lm

TARGET = time-ucc
SRC = time-ucc.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o

run: $(TARGET)
	mpirun -np 4 ./$(TARGET)

probe:
	@chmod +x probe_transports.sh
	@./probe_transports.sh

help:
	@echo "Available targets:"
	@echo "  make          - Build the benchmark"
	@echo "  make run      - Build and run with 4 processes"
	@echo "  make probe    - Probe available transports on this system"
	@echo "  make clean    - Remove built files"
	@echo "  make help     - Show this help message"

.PHONY: all clean run probe help

