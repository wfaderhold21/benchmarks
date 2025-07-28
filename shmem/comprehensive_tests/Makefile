CC = shmemcc
CFLAGS = -O2 -Wall -Wextra -I./common
LDFLAGS = 

# Directories
POINT_TO_POINT_DIR = point_to_point
ATOMIC_DIR = atomic
COLLECTIVE_DIR = collective
COMMON_DIR = common

# Executables
POINT_TO_POINT_BINS = $(POINT_TO_POINT_DIR)/shmem_put_bench $(POINT_TO_POINT_DIR)/shmem_get_bench
ATOMIC_BINS = $(ATOMIC_DIR)/shmem_atomic_bench
COLLECTIVE_BINS = $(COLLECTIVE_DIR)/shmem_collective_bench

ALL_BINS = $(POINT_TO_POINT_BINS) $(ATOMIC_BINS) $(COLLECTIVE_BINS)

# Default target
all: $(ALL_BINS)

# Point-to-point tests
$(POINT_TO_POINT_DIR)/shmem_put_bench: $(POINT_TO_POINT_DIR)/shmem_put_bench.c $(COMMON_DIR)/common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(POINT_TO_POINT_DIR)/shmem_get_bench: $(POINT_TO_POINT_DIR)/shmem_get_bench.c $(COMMON_DIR)/common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Atomic operations tests
$(ATOMIC_DIR)/shmem_atomic_bench: $(ATOMIC_DIR)/shmem_atomic_bench.c $(COMMON_DIR)/common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Collective operations tests
$(COLLECTIVE_DIR)/shmem_collective_bench: $(COLLECTIVE_DIR)/shmem_collective_bench.c $(COMMON_DIR)/common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Individual targets
point_to_point: $(POINT_TO_POINT_BINS)
atomic: $(ATOMIC_BINS)
collective: $(COLLECTIVE_BINS)

# Test targets
test: all
	@echo "=== Running SHMEM Comprehensive Benchmark Suite ==="
	@echo "Note: These tests require running with shmrun or mpirun"
	@echo ""
	@echo "Example usage:"
	@echo "  shmrun -np 4 $(POINT_TO_POINT_DIR)/shmem_put_bench"
	@echo "  shmrun -np 4 $(POINT_TO_POINT_DIR)/shmem_get_bench"
	@echo "  shmrun -np 4 $(ATOMIC_DIR)/shmem_atomic_bench"
	@echo "  shmrun -np 4 $(COLLECTIVE_DIR)/shmem_collective_bench"

test-put:
	@if command -v shmrun >/dev/null 2>&1; then \
		echo "Running SHMEM PUT benchmark..."; \
		shmrun -np 4 $(POINT_TO_POINT_DIR)/shmem_put_bench; \
	else \
		echo "shmrun not found. Please run manually: shmrun -np 4 $(POINT_TO_POINT_DIR)/shmem_put_bench"; \
	fi

test-get:
	@if command -v shmrun >/dev/null 2>&1; then \
		echo "Running SHMEM GET benchmark..."; \
		shmrun -np 4 $(POINT_TO_POINT_DIR)/shmem_get_bench; \
	else \
		echo "shmrun not found. Please run manually: shmrun -np 4 $(POINT_TO_POINT_DIR)/shmem_get_bench"; \
	fi

test-atomic:
	@if command -v shmrun >/dev/null 2>&1; then \
		echo "Running SHMEM Atomic operations benchmark..."; \
		shmrun -np 4 $(ATOMIC_DIR)/shmem_atomic_bench; \
	else \
		echo "shmrun not found. Please run manually: shmrun -np 4 $(ATOMIC_DIR)/shmem_atomic_bench"; \
	fi

test-collective:
	@if command -v shmrun >/dev/null 2>&1; then \
		echo "Running SHMEM Collective operations benchmark..."; \
		shmrun -np 4 $(COLLECTIVE_DIR)/shmem_collective_bench; \
	else \
		echo "shmrun not found. Please run manually: shmrun -np 4 $(COLLECTIVE_DIR)/shmem_collective_bench"; \
	fi

# Performance test with different PE counts
perf-test: all
	@echo "=== SHMEM Performance Test Suite ==="
	@for np in 2 4 8 16; do \
		echo ""; \
		echo "Testing with $$np PEs:"; \
		echo "------------------------"; \
		if command -v shmrun >/dev/null 2>&1; then \
			echo "PUT Test:"; \
			shmrun -np $$np $(POINT_TO_POINT_DIR)/shmem_put_bench --max-size 65536 --iterations 10000; \
			echo ""; \
			echo "GET Test:"; \
			shmrun -np $$np $(POINT_TO_POINT_DIR)/shmem_get_bench --max-size 65536 --iterations 10000; \
			echo ""; \
			echo "Atomic Test:"; \
			shmrun -np $$np $(ATOMIC_DIR)/shmem_atomic_bench --iterations 10000; \
			echo ""; \
			echo "Collective Test:"; \
			shmrun -np $$np $(COLLECTIVE_DIR)/shmem_collective_bench --max-size 16384 --iterations 1000; \
		else \
			echo "shmrun not found. Cannot run automated performance tests."; \
			break; \
		fi; \
	done

# Validation test
validation-test: all
	@echo "=== SHMEM Validation Test Suite ==="
	@if command -v shmrun >/dev/null 2>&1; then \
		echo "Running validation tests with 4 PEs..."; \
		echo "PUT validation:"; \
		shmrun -np 4 $(POINT_TO_POINT_DIR)/shmem_put_bench --max-size 1024 --iterations 100; \
		echo ""; \
		echo "GET validation:"; \
		shmrun -np 4 $(POINT_TO_POINT_DIR)/shmem_get_bench --max-size 1024 --iterations 100; \
		echo ""; \
		echo "Atomic validation:"; \
		shmrun -np 4 $(ATOMIC_DIR)/shmem_atomic_bench --iterations 1000; \
		echo ""; \
		echo "Collective validation:"; \
		shmrun -np 4 $(COLLECTIVE_DIR)/shmem_collective_bench --max-size 1024 --iterations 10; \
	else \
		echo "shmrun not found. Cannot run automated validation tests."; \
	fi

# Clean targets
clean:
	rm -f $(ALL_BINS)
	rm -f $(POINT_TO_POINT_DIR)/*.o $(ATOMIC_DIR)/*.o $(COLLECTIVE_DIR)/*.o
	rm -f core core.*

distclean: clean
	rm -f *~ $(POINT_TO_POINT_DIR)/*~ $(ATOMIC_DIR)/*~ $(COLLECTIVE_DIR)/*~ $(COMMON_DIR)/*~

# Help target
help:
	@echo "SHMEM Comprehensive Benchmark Suite"
	@echo "====================================="
	@echo ""
	@echo "Available targets:"
	@echo "  all              - Build all benchmarks"
	@echo "  point_to_point   - Build point-to-point benchmarks"
	@echo "  atomic           - Build atomic operations benchmarks"
	@echo "  collective       - Build collective operations benchmarks"
	@echo ""
	@echo "Test targets:"
	@echo "  test             - Show usage examples"
	@echo "  test-put         - Run PUT benchmark"
	@echo "  test-get         - Run GET benchmark"
	@echo "  test-atomic      - Run atomic operations benchmark"
	@echo "  test-collective  - Run collective operations benchmark"
	@echo "  perf-test        - Run comprehensive performance tests"
	@echo "  validation-test  - Run validation tests"
	@echo ""
	@echo "Maintenance targets:"
	@echo "  clean            - Remove binaries"
	@echo "  distclean        - Remove binaries and backup files"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  CC               - C compiler (default: shmemcc)"
	@echo "  CFLAGS           - C compiler flags (default: -O2 -Wall -Wextra)"
	@echo "  LDFLAGS          - Linker flags"

.PHONY: all point_to_point atomic collective test test-put test-get test-atomic test-collective perf-test validation-test clean distclean help 