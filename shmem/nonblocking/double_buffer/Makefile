CC = oshcc
CFLAGS = -O3 -Wall
LDFLAGS = -lm

TARGET = double_buffer
SRC = double_buffer.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean 