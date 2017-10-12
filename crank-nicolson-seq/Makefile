.PHONY: all clean

CC=gcc
CFLAGS=-Wall -Werror -Wextra -pedantic -std=c99

SRC=op-crank-nicolson-sequential

all:	$(SRC)

$(SRC):	$(SRC).c
	$(CC) $(CFLAGS) $< -o $@ -lm

clean:
	rm -f $(SRC)
