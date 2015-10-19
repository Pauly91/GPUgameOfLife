#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define nWorld 2

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
	// sleep(3);
	// system("clear");

}


int main(int argc, char const *argv[])
{
	if(argc != 3)
	{
		printf(" ./run [dataSet] [interations]\n");
		return -1;
	}
	int i,j,k,l;
	int x0,x1,y0,y1;
	int height = 0, width = 0;
	int checker;
	int interation = atoi(argv[2]);
	char ***world;
	FILE *fp = NULL;

    float time;

    cudaEvent_t start, stop;

	if((fp = fopen(argv[1],"r")) == NULL)
	{
		printf("Pattern file not opened\n");
		return -1;
	}
	fscanf(fp,"%d",&height);
	fscanf(fp,"%d",&width);

	world = (char ***) calloc(2,sizeof(char **)); // 0 for stage n ; 1 for stage n + 1
	for (i = 0; i < 2; ++i)
		world[i] = (char **) calloc(height,sizeof(char *));
	for (i = 0; i < 2; ++i)
	{
		for (j = 0; j < height; ++j)
		{
			world[i][j] = (char *) calloc(width,sizeof(char ));
		}
	
	}
	for (i = 0; i < height; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			fscanf(fp,"%d ",&world[0][i][j]);
			//printf("%d ",world[0][i][j]);
		}
		//printf("\n");
	}
	//printf("\n\n");
	fclose(fp);
    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );
    gpuErrchk( cudaEventRecord(start, 0) );

	for (i = 0,l = 0; i < interation; ++i)
	{
		for (j = 0; j < height; ++j)
		{
			for (k = 0; k < width; ++k)
			{
				x0 = (j - 1 + height)%height;
				x1 = (j + 1 + height)%height;
				y0 = (k - 1 + width)%width;
				y1 = (k + 1 + width)%width;
				//printf("x0:%d x1:%d y0:%d y1:%d l:%d\n",x0,x1,y0,y1,l);
				checker = world[l][x0][y0] + world[l][x0][k] + world[l][x0][y1] +
 				           world[l][j][y0] +                    world[l][j][y1] +
				          world[l][x1][y0] + world[l][x1][k] + world[l][x1][y1] ;
				//printf("%d\n",checker);
				if(checker == 3 || ( checker == 2 && world[l][j][k] == 0))
					world[!l][j][k] = 1;
				else
					world[!l][j][k] = 0;          
			}
		}
        //      display(world[!l],height,width);
        l = !l;
    }
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
    gpuErrchk( cudaEventElapsedTime(&time, start, stop) );  

	//display(world[!l],height,width);
	if((fp = fopen("CPUOutput","w")) == NULL)
	{
		printf("CPUOutput file not opened\n");
		return -1;
	}
	for (k = 0; k < nWorld; ++k)
	{
		for (i = 0; i < height; ++i)
		{
			for (j = 0; j < width; ++j)
			{
				fprintf(fp,"%d ",world[k][i][j]);
			}
			fprintf(fp,"\n");
		}
		fprintf(fp,"\n");
	}	
    printf("Time to generate using CPU:  %3.1f ms \n", time);	
    fclose(fp);


	return 0;


}