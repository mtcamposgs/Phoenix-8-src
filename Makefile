CC = gcc
CFLAGS = -Wall -Wextra -O3 -std=c99 -mavx2 -mfma -mpopcnt -mbmi -mbmi2 -fno-math-errno -fopenmp -flto
LDFLAGS = -lm -fopenmp -flto

ifeq ($(OS),Windows_NT)
LDFLAGS += -static
else ifneq (,$(findstring mingw,$(CC)))
LDFLAGS += -static
endif

TARGET = phoenix-windows-x86-64-avx2

SRCS = \
    main.c \
    uci.c \
    board.c \
    movegen.c \
    search.c \
    shra_eval.c \
    tt.c \
    zobrist.c

ASM_SRCS = weights_embed.S

OBJS = $(SRCS:.c=.o) $(ASM_SRCS:.S=.o)
HDRS = defs.h

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

weights_embed.o: weights_embed.S shra_weights.bin
	$(CC) -c weights_embed.S -o weights_embed.o

clean:
	@del /F /Q $(OBJS) $(TARGET) $(TARGET).exe 2>nul || rm -f $(OBJS) $(TARGET) $(TARGET).exe 2>/dev/null || true