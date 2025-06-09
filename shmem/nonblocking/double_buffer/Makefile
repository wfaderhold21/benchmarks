CC = oshcc
CFLAGS = -O3 -Wall
LDFLAGS = -lm

TARGET = double_buffer
TARGET_NOSYNC = double_buffer_nosync
SRC = double_buffer.c

all: $(TARGET) $(TARGET_NOSYNC)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

nosync: $(TARGET_NOSYNC)

$(TARGET_NOSYNC): $(SRC)
	$(CC) $(CFLAGS) -DWITHOUT_SYNC=1 -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET) $(TARGET_NOSYNC)

.PHONY: all clean nosync 