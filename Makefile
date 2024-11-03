src=main.cu kernel.cuh extras.cuh
nvcc=nvcc -Xcompiler -Wall --std=c++17 --ptxas-options=-v 

.PHONY: main

all: compile run

run: main
	./$< 127384954

compile: main

main: $(src)
	$(nvcc) $< -o $@
