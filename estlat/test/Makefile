BIN_DIR := ../bin

# Latency estimator.
EST := $(BIN_DIR)/estlat.py


SRCS := $(wildcard cities-*.txt)
OUTS := $(patsubst cities-%.txt, adjlst-%.txt, $(SRCS))

ALL := $(OUTS)


.PHONY: all clean

all: $(ALL)

clean:
	@rm -f $(ALL)


# Compute adjacency list (containing latencies between the vertices) comprising
# pairwise edges between all cities.
adjlst-%.txt: cities-%.txt
	@$(EST) $< $@
