# Makefile for CG Poisson 3D OpenSHMEM solver
# 
# Usage:
#   make                    # Compile without debug output (default)
#   make DEBUG=1           # Compile with debug output enabled
#   make clean             # Clean build files

CC = oshcc
CFLAGS = -O2 -Wall
LDFLAGS = -lm
TARGET = cg_poisson_3d_oshmem
SOURCE = cg_poisson_3d_oshmem.c

# Default target (no debug)
$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)

# Debug target
debug: $(SOURCE)
	$(CC) $(CFLAGS) -DDEBUG=1 -o $(TARGET) $(SOURCE) $(LDFLAGS)

# Clean target
clean:
	rm -f $(TARGET)

# Show help
help:
	@echo "Available targets:"
	@echo "  make          - Compile without debug output"
	@echo "  make debug    - Compile with debug output enabled"
	@echo "  make clean    - Remove compiled binary"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Example usage:"
	@echo "  make DEBUG=1"
	@echo "  ./cg_poisson_3d_oshmem 200 200 200"

.PHONY: clean help debug 
