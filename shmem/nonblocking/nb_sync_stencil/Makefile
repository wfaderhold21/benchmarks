CC      = shmemcc
CFLAGS  = -O3 -Wall
LDFLAGS = -lm

TARGET = nb_sync_stencil
SRC    = nb_sync_stencil.c

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: clean
