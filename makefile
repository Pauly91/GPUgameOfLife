all:
	nvcc -g -G gpuGameOfLife.cu -o gpu -lm -Xptxas -v 
	nvcc -g -G gpuGameOfLifeCharAccess.cu -o gpuChar -lm -Xptxas -v 
	nvcc -g -G cpuGameOfLife.cu -o cpu
clean:
	rm gpu
	rm cpu 
	rm gpuChar
