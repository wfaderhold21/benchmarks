# tools/icmc -- offline parser tests for the ICMC counter/QP parsers
#
#   cc -O2 test_parsers && ./test_parsers      (no NIC, no root)
#   bash qpsweep.sh send <host> ...            (on a spr/emr node, unprivileged)

CC?=cc
CFLAGS?=-O2 -Wall -Wextra
COMMON=../../common

UNAME := $(shell uname)

.PHONY: all test clean

ifeq ($(UNAME),Linux)
all: test_parsers raw_iio
else
all: test_parsers
endif

test_parsers: test_parsers.c $(COMMON)/qp_count.c $(COMMON)/iio_pmu.c \
              $(COMMON)/qp_count.h $(COMMON)/iio_pmu.h
	$(CC) $(CFLAGS) -std=c11 -I$(COMMON) test_parsers.c \
	    $(COMMON)/qp_count.c $(COMMON)/iio_pmu.c -o $@

# raw_iio: open the IIO read counters exactly like iio_pmu.c and print the
# deltas over <secs>; cross-check against `perf stat` on a spr/emr node.
# Requires Linux; links nothing beyond libc (perf_event_open).
raw_iio: raw_iio.c
	$(CC) $(CFLAGS) -std=c11 raw_iio.c -o $@

test: test_parsers
	./test_parsers
clean:
	rm -f test_parsers raw_iio
