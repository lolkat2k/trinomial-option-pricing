C=nvcc
C_FLAGS=-O3

FILE=trinom-cuda

.PHONY: all clean

all:		$(FILE)

$(FILE):	$(FILE).cu
		$(C) $(C_FLAGS) $< -o $@ -lm

clean:
		rm -f  $(FILE)
