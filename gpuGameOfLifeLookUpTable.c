#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#define nWorld 2
#define N 24

#define cellWidth 6
#define cellHeight 3
#define nextGenWidth 4

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void display(char **world,int height, int width)
{
	int i,j;
	for (i = 0; i < height; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			printf("%d ",world[i][j]);
		}
		printf("\n");
	}
	sleep(3);
	system("clear");

}
__global__ void worldGenerator(char *world, int height, int width, int swap, char *lookUp)
{

    const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                            blockIdx.y * blockDim.y + threadIdx.y);

    const int thread_1D_pos = thread_2D_pos.y * width + thread_2D_pos.x;
 	

    int z = (swap)*height*width;



    
    
    int checker = 0;

	int x = 0;
	int y = 0;

	int X = blockIdx.x * blockDim.x + threadIdx.x * nextGenWidth  + width; //  multiply with nextGenWidth to go to the next 4 ceel
	int Y = blockIdx.y * blockDim.y + threadIdx.y + height;

	char mask = 1; // make this as shared or const
	char newGen = 0;

	for (int i = -1; i < cellHeight - 1; ++i)
	{
		for (int j = -1; j < cellWidth - 1; ++j)
		{
			x = (X + j)%width;
			y = (Y + i)%height;

			if(world[z + y * width + x] == 1)
			{
				checker |= mask;
				checker <<= 1;
			}

		}
	}


	newGen = lookUp[checker]; // need to create this table - do it back home 

    for (int i = 0; i < nextGenWidth; ++i)
    {
    	
    	world[(!swap)*height*width +  Y * width + X + i] = newGen & mask;
    	newGen >>= 1;
    }    

}

// __global__ void worldGenerator(int height,int width)
// {
// 	printf("height:%d width:%d\n",height,width);
// }

int main(int argc, char const *argv[])
{
	if(argc != 3)
	{
		printf(" ./run [dataSet] [interations]\n");
		return -1;
	}
	int i,j,k,swap;
	int height = 0, width = 0;
	int iteration = atoi(argv[2]);
	char *world = NULL;
	char *d_world = NULL;
	char remark[50] = "No shared mem";

    float timez;

    cudaEvent_t start, stop;

// Read the pattern

	FILE *fp = NULL;
	if((fp = fopen(argv[1],"r")) == NULL)
	{
		printf("Pattern file not opened\n");
		return -1;
	}


	fscanf(fp,"%d",&height);
	fscanf(fp,"%d",&width);

// allocate memory	
	world = (char *) calloc(nWorld*width*height,sizeof(char));
	for (i = 0; i < height; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			fscanf(fp,"%d ",&world[0 * width * height + i * width + j]);
		}
	}
	fclose(fp);
// Getting the Data
	// for (k = 0; k < nWorld; ++k)
	// {
	// 	for (i = 0; i < height; ++i)
	// 	{
	// 		for (j = 0; j < width; ++j)
	// 		{
	// 			printf("%d ",world[k * width * height + i * width + j]);
	// 		}
	// 		printf("\n");
	// 	}
	// 	printf("\n");
	// }

// GPU sections
    dim3 grid(height/N,width/N);            // defines a grid of 256 x 1 x 1 blocks
    dim3 block(N,N); 	


    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );
    gpuErrchk( cudaEventRecord(start, 0) );

	gpuErrchk(cudaMalloc((void**)&d_world, nWorld * width * height * sizeof(char)));
	gpuErrchk(cudaMemcpy(d_world, world, nWorld * width * height * sizeof(char), cudaMemcpyHostToDevice));

	for (i = 0,swap = 0; i < iteration; ++i)
	{
        worldGenerator<<< grid, block>>>(d_world,height,width,swap);
		swap = !swap;
	}

	gpuErrchk(cudaMemcpy(world, d_world, nWorld * width * height * sizeof(char), cudaMemcpyDeviceToHost));
	

    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
    gpuErrchk( cudaEventElapsedTime(&timez, start, stop) );  
	
	if((fp = fopen("GPUOutputChar","w")) == NULL)
	{
		printf("GPUOutput file not opened\n");
		return -1;
	}
	for (k = 0; k < nWorld; ++k)
	{
		for (i = 0; i < height; ++i)
		{
			for (j = 0; j < width; ++j)
			{
				fprintf(fp,"%d ",world[k * width * height + i * width + j]);
			}
			fprintf(fp,"\n");
		}		
		fprintf(fp,"\n");
	}	
    printf("Time to generate using GPU:  %3.1f ms \n", timez);	
 	if((fp = fopen("timing","a+")) == NULL)
	{
		printf("GPUOutput file not opened\n");
		return -1;
	}   
	fprintf(fp, "GPU char access with %s- > Height: %d Width: %d Iteration: %d Time: %f\n",remark,height,width,iteration,timez);
	return 0;
}

// algorithm to try out
/*

Since int access takes in 4 bytes use that information to process 3 cells
ie push data to device as char but read them as int



*/