# Makefile for UCC Context/Team Creation Benchmark

CC = mpicc
HPCX = /work/nvidia/hpcx-v2.22-gcc-doca_ofed-ubuntu24.04-cuda12-x86_64
UCC_DIR = $(HPCX)/ucc
UCX_DIR = $(HPCX)/ucx
CFLAGS = -Wall -O2 -g -I$(UCC_DIR)/include -I../../common
LDFLAGS = -L$(UCX_DIR)/lib -L$(UCC_DIR)/lib -lucc -lucs -lm

TARGET = time-ucc
SRC = time-ucc.c

all: $(TARGET)

$(TARGET): $(SRC) ../../common/transport_detect.o
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) ../../common/transport_detect.o $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o ../../common/transport_detect.o

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

