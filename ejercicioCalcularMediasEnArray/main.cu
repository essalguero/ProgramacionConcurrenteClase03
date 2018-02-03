#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

#define MEM_DIM 64
#define RADIO 3
#define SIZE_BLOQUE 8
#define SIZE_GRID 8
#define RADIO 3


/*Programar una funcion que haga la media de numeros end GPU sin memoria compartida, en GPU con memoria compartida
  y en CPU. Comparar los tiempos de ejecucion*/

__global__ void kernel_Shared(int *d_input, int *d_output)
{
	int i;
	int valorFinal = 0;
	__shared__ int arrayValores[ MEM_DIM + RADIO + RADIO ];

	//Inicializar el array para poder calcular las medias
	arrayValores[threadIdx.x + RADIO] = 0;

	//Inicializar las posiciones extras en el array
	if (threadIdx.x < RADIO)
		arrayValores[threadIdx.x] = 0;

	if (threadIdx.x >= (SIZE_BLOQUE - RADIO))
		arrayValores[threadIdx.x + RADIO] = 0;

/*
	//En esta posicion los valores de arrayValores son correctos - Inicializados a 0
	for(int i = 0; i < blockDim.x + RADIO + RADIO; ++i)
	{
		printf("Valor deberia ser 0: %d\n", arrayValores[i]);
	}
*/

	// Sincronizar todos los threads - Se puede omitir?
	__syncthreads();
	
	//Copiar los valores desde la memoria global a la memoria compartida
	arrayValores[threadIdx.x + RADIO] = d_input[blockIdx.x * blockDim.x + threadIdx.x];

	// 
	/*if (threadIdx.x == 0)
	{
		for(int i = 0; i < blockDim.x + RADIO + RADIO; ++i)
		{
			printf("Valor deberia ser 0: %d\n", arrayValores[i]);
		}
	}*/

	
	//d_output[blockIdx.x * blockDim.x + threadIdx.x];
	


	/*if (threadIdx.x == 0)
	{
		for(int i = 0; i < MEM_DIM + RADIO + RADIO; ++i)
			printf(" %d", arrayValores[i]);
	}
	printf("\n");*/

	

	//Copiar los valores extras
	if (threadIdx.x < RADIO)
	{
		if (blockIdx.x > 0)
		{
			arrayValores[threadIdx.x] = d_input[(blockIdx.x * blockDim.x + threadIdx.x) - RADIO];
		}
	}

	

	if (threadIdx.x >= (SIZE_BLOQUE - RADIO))
	{
		 if (blockIdx.x < SIZE_GRID - 1)
		{
			arrayValores[threadIdx.x + RADIO + RADIO] = d_input[(blockIdx.x * blockDim.x + threadIdx.x) + RADIO];
		}
	}


	if (threadIdx.x == 0)
	{
		for(int i = 0; i < blockDim.x + RADIO + RADIO; ++i)
		{
			printf("Valor kernel (%d, %d): %d\n", blockIdx.x, i, arrayValores[i]);
		}
		printf("%d\n\n", blockIdx.x * blockDim.x + threadIdx.x);
	}



	//Sincronizar los threads
	__syncthreads();

	//Hacer la media en el array de outputs
	for (i = -RADIO; i <= RADIO; ++i)
	{
		valorFinal += arrayValores[(threadIdx.x + RADIO) + i];
	}

	valorFinal /= (RADIO + RADIO + 1);

	printf("Valor en el thread actual (%d, %d): %d\n", blockIdx.x, threadIdx.x, valorFinal);

	d_output[blockIdx.x * blockDim.x + threadIdx.x] = valorFinal;

printf("Bloque: %d -> Thread: %d -> PosicionArray: %d -> Posicion Array Global: %d -> Valor Guardado: %d\n", blockIdx.x, threadIdx.x, threadIdx.x + RADIO, blockIdx.x * blockDim.x + threadIdx.x, arrayValores[threadIdx.x + RADIO]);

}

double tiempo( void )
{
	struct timeval  tv;
	gettimeofday(&tv, NULL);

	return (double) (tv.tv_usec) / 1000000 + (double) (tv.tv_sec);
}

int main(int argc, char** argv)
{
	double tiempoInicio;
	double tiempoFin;
	
	int n = SIZE_BLOQUE * SIZE_GRID;
	

	printf("\nElementos a reservar: %d\n\n\n", n);

	int numBytes = n * sizeof(int);

	int *d_input;
	int *d_output;

	int *h_input;
	int *h_output;

	cudaMalloc((void **) &d_input, numBytes );

	if(cudaSuccess != cudaGetLastError())
	{
		printf("Error de cuda\n");
	}

	cudaMalloc((void **) &d_output, numBytes );

	if(cudaSuccess != cudaGetLastError())
	{
		printf("Error de cuda\n");
	}

	cudaMemset(d_output, 0, n);

	if(cudaSuccess != cudaGetLastError())
	{
		printf("Error de cuda\n");
	}


	h_input = (int *)malloc(numBytes);
	h_output = (int *)malloc(numBytes);

	for(int i = 0; i < n; ++i)
		h_input[i] = i;

	cudaMemcpy (d_input, h_input, numBytes, cudaMemcpyHostToDevice);

	if(cudaSuccess != cudaGetLastError())
	{
		printf("Error de cuda\n");
	}

	


	dim3 blockSize(SIZE_BLOQUE);
	dim3 gridSize(SIZE_GRID);


	tiempoInicio = tiempo();
	kernel_Shared <<<gridSize, blockSize>>>(d_input, d_output);
	cudaThreadSynchronize();

	if(cudaSuccess != cudaGetLastError())
	{
		printf("Error de cuda _1\n");
	}

	tiempoFin = tiempo();
	
	
	printf("Tiempo de inicio Kernel: %lf\n", tiempoInicio);
	printf("Tiempo de fin Kernel: %lf\n", tiempoFin);
	printf("Tiempo total: %lf\n\n\n", tiempoFin - tiempoInicio);


	tiempoInicio = tiempo();
	cudaMemcpy (h_output, d_output, numBytes, cudaMemcpyDeviceToHost);
	tiempoFin = tiempo();

	if ( cudaSuccess != cudaGetLastError() )
		printf( "Error! _2\n" );

	printf("Tiempo de inicio Transferencia: %lf\n", tiempoInicio);
	printf("Tiempo de fin Transferencia: %lf\n", tiempoFin);
	printf("Tiempo total: %lf\n\n\n", tiempoFin - tiempoInicio);


	for(int i = 0; i < n; ++i)
	{
		printf("%d - ", h_output[i]);
		
	}
	printf("\n\n\nDone.\n");

	return 0;
}
