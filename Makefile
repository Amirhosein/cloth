# ===== Build settings (macOS-friendly) =====
# On macOS, "gcc" is clang; use clang explicitly.
CC       := clang
STD      := -std=c11
WARN     := -Wall -Wextra -Wpedantic
THREADS  := -pthread
PKG_GSL_CFLAGS := $(shell pkg-config --cflags gsl)
PKG_GSL_LIBS   := $(shell pkg-config --libs gsl)

# DEBUG=1 enables ASan/UBSan + symbols; DEBUG=0 for optimized release
DEBUG    ?= 1

ifeq ($(DEBUG),1)
  OPTS   := -O0 -g
  SAN    := -fsanitize=address,undefined -fno-omit-frame-pointer
else
  OPTS   := -O2
  SAN    :=
endif

CFLAGS   := $(STD) $(WARN) $(THREADS) $(OPTS) $(SAN) $(PKG_GSL_CFLAGS)
LDFLAGS  := $(THREADS) $(SAN) $(PKG_GSL_LIBS)

SRC := \
  ./src/cloth.c \
  ./src/heap.c \
  ./src/array.c \
  ./src/list.c \
  ./src/event.c \
  ./src/payments.c \
  ./src/htlc.c \
  ./src/routing.c \
  ./src/network.c \
  ./src/utils.c

BIN := cloth

.PHONY: all build run clean

all: build

build: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Example: make run SEED=1992 OUT=./out
run: build
	./run-simulation.sh $(SEED) $(OUT)

clean:
	rm -f $(BIN)
