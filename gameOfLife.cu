#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#define nWorld 2
#define N 24

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
__global__ void worldGenerator(int *world, int height, int width, int swap)
{

    const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                            blockIdx.y * blockDim.y + threadIdx.y);

    const int thread_1D_pos = thread_2D_pos.y * width + thread_2D_pos.x;

    // if(thread_1D_pos == 0)
    // {
    // 	for (int k = 0; k < nWorld; ++k)
    // 	{
    // 		for (int i = 0; i < height; ++i)
    // 		{
    // 			for (int j = 0; j < width; ++j)
    // 			{
    // 				printf("%d ",world[k * height * width + i * width + j]);
    // 			}
    // 			printf("\n");
    // 		}
    // 		printf("\n\n");
    // 	}
    // }
    
    int z = (swap)*height*width;
    int checker;
    //printf("%d\n",swap);
    //printf("x:%d y:%d\n",thread_2D_pos.y,thread_2D_pos.x);
	int x0 = (thread_2D_pos.y - 1 + height)%height;
	int x1 = (thread_2D_pos.y + 1 + height)%height;
	int y0 = (thread_2D_pos.x - 1 + width)%width;
	int y1 = (thread_2D_pos.x + 1 + width)%width;
	//printf("x0:%d x1:%d y0:%d y1:%d \n",x0 * width,x1 * width ,y0,y1);
	// printf("%d:%d %d:%d %d:%d \n%d:%d %d:%d %d:%d \n%d:%d %d:%d %d:%d \n\n",z + x0 * width + y0,world[z + x0 * width + y0],                z + x0 * width + thread_2D_pos.x,world[z + x0 * width + thread_2D_pos.x],                z + x0 * width + y1,world[z + x0 * width + y1],
	// 							   z + thread_2D_pos.y * width + y0,world[z + thread_2D_pos.y * width + y0],   z + thread_2D_pos.y * width + thread_2D_pos.x,world[z + thread_2D_pos.y * width + thread_2D_pos.x],   z + thread_2D_pos.y * width + y1,world[z + thread_2D_pos.y * width + y1],
	// 							                z + x1 * width + y0,world[z + x1 * width + y0],                z + x1 * width + thread_2D_pos.x,world[z + x1 * width + thread_2D_pos.x],                z + x1 * width + y1,world[z + x1 * width + y1]);

	checker =              world[z + x0 * width + y0] + world[z + x0 * width + thread_2D_pos.x] +   world[z + x0 * width + y1] +
		      world[z + thread_2D_pos.y * width + y0] +                                world[z + thread_2D_pos.y * width + y1] +
	                       world[z + x1 * width + y0] + world[z + x1 * width + thread_2D_pos.x] +   world[z + x1 * width + y1] ;
	//printf("%d\n",checker);
	if(checker == 3 || ( checker == 2 && world[(swap)*height*width +  thread_1D_pos] == 0))
		world[(!swap)*height*width +  thread_1D_pos] = 1;
	else
		world[(!swap)*height*width +  thread_1D_pos] = 0;          

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
	int interation = atoi(argv[2]);
	int *world = NULL;
	int *d_world = NULL;

    float time;

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
	world = (int *) calloc(nWorld*width*height,sizeof(int));
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

	gpuErrchk(cudaMalloc((void**)&d_world, nWorld * width * height * sizeof(int)));
	gpuErrchk(cudaMemcpy(d_world, world, nWorld * width * height * sizeof(int), cudaMemcpyHostToDevice));

	for (i = 0,swap = 0; i < interation; ++i)
	{
        worldGenerator<<< grid, block>>>(d_world,height,width,swap);
		swap = !swap;
	}

	gpuErrchk(cudaMemcpy(world, d_world, nWorld * width * height * sizeof(int), cudaMemcpyDeviceToHost));
	

    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
    gpuErrchk( cudaEventElapsedTime(&time, start, stop) );  
	
	if((fp = fopen("GPUOutput","w")) == NULL)
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
    printf("Time to generate:  %3.1f ms \n", time);	
	return 0;
}

// algorithm to try out
/*

Since int access takes in 4 bytes use that information to process 3 cells
ie push data to device as char but read them as int



*/