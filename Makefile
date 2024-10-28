src=main.cu kernel.cuh extras.cuh
nvcc=nvcc -Xcompiler -Wall

.PHONY: main

main: $(src)
	$(nvcc) $< -o $@
