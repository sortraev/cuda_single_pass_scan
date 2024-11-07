kernel=../kernels/kernel.cuh
src=bench_main.cu ../util.cuh ../host.cuh $(kernel) ../extras.cuh
nvcc=nvcc -I../kernels -I.. -Xcompiler -Wall --std=c++17 --ptxas-options=-v 

.PHONY: bench_main

all: compile run

compile: bench_main

run: bench_main
	./$< 1337


bench_main: $(src)
	$(nvcc) $< -o $@
	# $(nvcc) $< -D__B=256 -D__Q=13 -o $@

clean:
	rm -rf bench_main
