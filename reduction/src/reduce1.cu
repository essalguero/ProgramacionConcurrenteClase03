/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

////////////////////////////////////////////////////////////////////////////////
// Includes

#include <stdio.h>
#include <cutil.h>

////////////////////////////////////////////////////////////////////////////////
// Static variable declarations

// Initial execution configuration
#define NUM_THREADS_PER_BLOCK       512
#define NUM_BLOCKS                  4
#define NUM_THREADS                 (NUM_THREADS_PER_BLOCK * NUM_BLOCKS)

// Result buffers
static int* d_Result[2];

////////////////////////////////////////////////////////////////////////////////
// Forward declarations

__global__ void reduce_kernel(const int*, unsigned int, int, int*);
int adjustExecutionConfiguration(int&, int&, unsigned int);

////////////////////////////////////////////////////////////////////////////////
//! Allocate any necessary device memory.
////////////////////////////////////////////////////////////////////////////////
extern "C" void reduceAllocate()
{
    for (int i = 0; i < 2; ++i)
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_Result[i],
                                  NUM_THREADS * sizeof(int)));
}

////////////////////////////////////////////////////////////////////////////////
//! Free device memory.
////////////////////////////////////////////////////////////////////////////////
extern "C" void reduceFree()
{
    for (int i = 0; i < 2; ++i)
        CUDA_SAFE_CALL(cudaFree(d_Result[i]));
}

////////////////////////////////////////////////////////////////////////////////
//! Compute the sum of all values from the input array through parallel
//! reduction.
//! @param values       input array of values
//! @param numValues    number of values
////////////////////////////////////////////////////////////////////////////////
extern "C" int* reduce(const int* values, unsigned int numValues)
{
    // Execution configuration
    int numThreadsPerBlock = NUM_THREADS_PER_BLOCK;
    int numBlocks = NUM_BLOCKS;
    int numThreads = numBlocks * numThreadsPerBlock;

    // The first pass reduces the input array to an array of size equal to the
    // total number of threads in the grid
    int numValuesPerThread = (int)ceilf(numValues / (float)numThreads);

    //printf("Numero bloques \t numero hilos por bloque \t numero valores\n");
    //printf("%d \t %d \t %d \t %d\n", numBlocks, numThreadsPerBlock, numValues, numValuesPerThread);
    reduce_kernel<<<numBlocks, numThreadsPerBlock>>>
                   (values, numValues, numValuesPerThread, d_Result[0]);
    CUT_CHECK_ERROR("Kernel execution failed");
    
    // The subsequent passes reduce the array by half each pass
    numValues = numThreads;
    int i = 0;
    while (numValues > 1) {
    
        // Adjust execution configuration depending on current array size
        numThreads = adjustExecutionConfiguration(numBlocks, numThreadsPerBlock,
                                                  numValues);
        
        // Reduces array by half
        numValuesPerThread = (int)ceilf(numValues / (float)numThreads);
	
        //printf("%d \t %d \t %d \t %d\n", numBlocks, numThreadsPerBlock, numValues, numValuesPerThread);
	
        reduce_kernel<<<numBlocks, numThreadsPerBlock>>>
                       (d_Result[i], numValues, numValuesPerThread, d_Result[!i]);
        CUT_CHECK_ERROR("Kernel execution failed");
        numValues /= 2;
        i = !i;
    }
    
    return d_Result[i];
}

////////////////////////////////////////////////////////////////////////////////
//! Reduce an array of input values
//! @param valuesIn        array of input values
//! @param numValues       number of input values
//! @param valuesOut       array of reduced values
////////////////////////////////////////////////////////////////////////////////
__global__ void reduce_kernel(const int* valuesIn, unsigned int numValues,
                              int numValuesPerThread,
                              int* valuesOut)
{
    // Execution configuration
    int numThreadsPerBlock = blockDim.x;
    
    // Index in the grid
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    //int index = 0 /* TODO */;
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Each thread processes numValuesPerThread values
    int result = 0;
    //int begin = min(0 /* TODO */, numValues);
    //int end = min(0 /* TODO */, numValues);
    int begin = min(index * numValuesPerThread, numValues);
    int end = min(begin + numValuesPerThread, numValues);
    for (int i = begin; i < end; ++i)
        result += valuesIn[i];
    
    // Each thread writes the result of the reduction to device memory
    //valuesOut[0 /* TODO */] = result;
    valuesOut[index] = result;
}

////////////////////////////////////////////////////////////////////////////////
//! Adjust execution configuration so that there are fewer threads than elements
//! in the array of values to be reduced. That way, each thread has at least two
//! elements to add together.
//! @param numBlocks     number of blocks
//! @param dataOut       number of threads per block
//! @param numValues     number of remaining values that need to be reduced
////////////////////////////////////////////////////////////////////////////////
int adjustExecutionConfiguration(int& numBlocks, int& numThreadsPerBlock,
                                 unsigned int numValues)
{
    // Total number of threads in the grid
    int numThreads = numBlocks * numThreadsPerBlock;
    
    // Divide by two first the number of threads per block and then the number of
    // blocks until there are fewer threads than elements in the array of values
    // to be reduced
    while (numValues <= numThreads) {
        numThreadsPerBlock /= 2;
        if (numThreadsPerBlock < 1) {
            numThreadsPerBlock = 1;
            numBlocks /= 2;
        }
        numThreads = numBlocks * numThreadsPerBlock;
    }
    
    return numThreads;
}
