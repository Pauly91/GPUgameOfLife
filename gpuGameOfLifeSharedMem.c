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
__global__ void worldGenerator(char *world, int height, int width, int swap)
{

    const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                            blockIdx.y * blockDim.y + threadIdx.y);

    const int thread_1D_pos = thread_2D_pos.y * width + thread_2D_pos.x;
 	
    __shared__ char sWorld[N*N + 1]; // no zero padding

    int z = (swap)*height*width;

    sWorld[threadIdx.y * blockDim.x + threadIdx.x] = world[z +  thread_1D_pos];

    __syncthreads();

    
    
    int checker = 0;
    //printf("%d\n",swap);
    //printf("x:%d y:%d\n",thread_2D_pos.y,thread_2D_pos.x);






	
	//printf("x0:%d x1:%d y0:%d y1:%d \n",x0 * width,x1 * width ,y0,y1);
	// printf("%d:%d %d:%d %d:%d \n%d:%d %d:%d %d:%d \n%d:%d %d:%d %d:%d \n\n",z + x0 * width + y0,world[z + x0 * width + y0],                z + x0 * width + thread_2D_pos.x,world[z + x0 * width + thread_2D_pos.x],                z + x0 * width + y1,world[z + x0 * width + y1],
	// 							   z + thread_2D_pos.y * width + y0,world[z + thread_2D_pos.y * width + y0],   z + thread_2D_pos.y * width + thread_2D_pos.x,world[z + thread_2D_pos.y * width + thread_2D_pos.x],   z + thread_2D_pos.y * width + y1,world[z + thread_2D_pos.y * width + y1],
	// 							                z + x1 * width + y0,world[z + x1 * width + y0],                z + x1 * width + thread_2D_pos.x,world[z + x1 * width + thread_2D_pos.x],                z + x1 * width + y1,world[z + x1 * width + y1]);

	if(threadIdx.x == 0 || threadIdx.x == (blockDim.x - 1) || threadIdx.y == 0 || threadIdx.y == (blockDim.y - 1) )
	{
		

		int x0 = (thread_2D_pos.y - 1 + height)%height;
		int x1 = (thread_2D_pos.y + 1 + height)%height;
		int y0 = (thread_2D_pos.x - 1 + width)%width;
		int y1 = (thread_2D_pos.x + 1 + width)%width;



		checker =                   world[z + x0 * width + y0] +  world[z + x0 * width + thread_2D_pos.x] +    world[z + x0 * width + y1] +
				      world[z + thread_2D_pos.y * width + y0] +                                       world[z + thread_2D_pos.y * width + y1] +
			                        world[z + x1 * width + y0] +  world[z + x1 * width + thread_2D_pos.x] +    world[z + x1 * width + y1] ;		
	
		char holder = world[(swap)*height*width +  thread_1D_pos];
	}
	else
	{	
		
		int Sx0 = threadIdx.y - 1;
		int Sx1 = threadIdx.y + 1;
		int Sy0 = threadIdx.x - 1;
		int Sy1 = threadIdx.x + 1;

		checker =           sWorld[Sx0 * blockDim.x + Sy0] +  sWorld[Sx0 * blockDim.x + threadIdx.x] +      sWorld[Sx0 * blockDim.x + Sy1] +
		           sWorld[threadIdx.y * blockDim.x + Sy0] +                                             sWorld[threadIdx.y * blockDim.x + Sy1] +
	                        sWorld[Sx1 * blockDim.x + Sy0] +  sWorld[Sx1 * blockDim.x + threadIdx.x] +      sWorld[Sx1 * blockDim.x + Sy1] ;
	
		char holder = sWorld[threadIdx.y * blockDim.x + threadIdx.x];
	}





	// checker =              (int) world[z + x0 * width + y0] + (int) world[z + x0 * width + thread_2D_pos.x] +   (int) world[z + x0 * width + y1] +
	// 	      (int )world[z + thread_2D_pos.y * width + y0] +                                (int) world[z + thread_2D_pos.y * width + y1] +
	//                        (int) world[z + x1 * width + y0] + (int) world[z + x1 * width + thread_2D_pos.x] +   (int) world[z + x1 * width + y1] ;
	// //printf("%d\n",checker);
	if(checker == 3 || ( checker == 2 && holder == 0))
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
	int iteration = atoi(argv[2]);
	char *world = NULL;
	char *d_world = NULL;
	char remark[50] = "With shared memory";

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


/*

Random access use texture memory - https://www.olcf.ornl.gov/tutorials/cuda-game-of-life/
Use of better block size

*/