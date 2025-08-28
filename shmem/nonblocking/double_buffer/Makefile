CC = shmemcc
CFLAGS = -O3 -Wall
LDFLAGS = -lm

TARGET = double_buffer
TARGET_NOSYNC = double_buffer_nosync
TARGET_BLOCKING = double_buffer_blocking
TARGET_BCAST = double_buffer_bcast
TARGET_BCAST_NOSYNC = double_buffer_bcast_nosync
TARGET_BCAST_BLOCKING = double_buffer_bcast_blocking
SRC = double_buffer.c
BCAST_SRC = double_buffer_bcast.c

all: $(TARGET) $(TARGET_NOSYNC) $(TARGET_BLOCKING) $(TARGET_BCAST) $(TARGET_BCAST_NOSYNC) $(TARGET_BCAST_BLOCKING)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

nosync: $(TARGET_NOSYNC)

blocking: $(TARGET_BLOCKING)

$(TARGET_NOSYNC): $(SRC)
	$(CC) $(CFLAGS) -DWITHOUT_SYNC=1 -o $@ $< $(LDFLAGS)

$(TARGET_BLOCKING): $(SRC)
	$(CC) $(CFLAGS) -DWITH_BLOCKING=1 -o $@ $< $(LDFLAGS)

$(TARGET_BCAST): $(BCAST_SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

nosync: $(TARGET_BCAST_NOSYNC)

blocking: $(TARGET_BCAST_BLOCKING)

$(TARGET_BCAST_NOSYNC): $(BCAST_SRC)
	$(CC) $(CFLAGS) -DWITHOUT_SYNC=1 -o $@ $< $(LDFLAGS)

$(TARGET_BCAST_BLOCKING): $(BCAST_SRC)
	$(CC) $(CFLAGS) -DWITH_BLOCKING=1 -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET) $(TARGET_NOSYNC) $(TARGET_BLOCKING) $(TARGET_BCAST) $(TARGET_BCAST_NOSYNC) $(TARGET_BCAST_BLOCKING)

.PHONY: all clean nosync 
