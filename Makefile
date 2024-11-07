kernel=kernels/kernel.cuh
src=main.cu util.cuh host.cuh $(kernel) extras.cuh

nvcc=nvcc -Ikernels -Xcompiler -Wall --std=c++17 --ptxas-options=-v

.PHONY: main

all: compile run

run: main
	./$< 127384954

compile: main

main: $(src)
	$(nvcc) $< -o $@
