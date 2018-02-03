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

// Result buffer
static int* d_Result;

////////////////////////////////////////////////////////////////////////////////
// Forward declarations

__global__ void reduce_kernel(const int*, unsigned int, int*);

////////////////////////////////////////////////////////////////////////////////
//! Allocate any necessary device memory.
////////////////////////////////////////////////////////////////////////////////
extern "C" void reduceAllocate()
{
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_Result, NUM_BLOCKS * sizeof(int)));
}

////////////////////////////////////////////////////////////////////////////////
//! Free device memory.
////////////////////////////////////////////////////////////////////////////////
extern "C" void reduceFree()
{
    CUDA_SAFE_CALL(cudaFree(d_Result));
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

    // The first pass reduces the input array to an array of size equal to the
    // total number of blocks in the grid
    int sharedMemorySize = numThreadsPerBlock * sizeof(int);
    reduce_kernel<<<numBlocks, numThreadsPerBlock, sharedMemorySize>>>
                   (values, numValues, d_Result);
    CUT_CHECK_ERROR("Kernel execution failed");
    
    // The second pass launches only one block to perform the final reduction
    numThreadsPerBlock = numBlocks;
    numBlocks = 1;
    sharedMemorySize = numThreadsPerBlock * sizeof(int);
    reduce_kernel<<<numBlocks, numThreadsPerBlock, sharedMemorySize>>>
                   (d_Result, numThreadsPerBlock, d_Result);
    
    return d_Result;
}

////////////////////////////////////////////////////////////////////////////////
//! Reduce an array of input values
//! @param valuesIn        array of input values
//! @param numValues       number of input values
//! @param valuesOut       array of reduced values
////////////////////////////////////////////////////////////////////////////////
__global__ void reduce_kernel(const int* valuesIn, unsigned int numValues,
                              int* valuesOut)
{
    // Execution configuration
    int numBlocks = gridDim.x;
    int numThreadsPerBlock = blockDim.x;
    int numThreads = numBlocks * numThreadsPerBlock;
    
    // Index in the grid
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int index = blockID * numThreadsPerBlock + threadID;
    
    // Each thread processes as many values as necessary with a stride equal to
    // the total number of threads in the grid and stores the result in shared
    // memory
    extern __shared__ int sresult[];
    sresult[threadID] = 0;
    for (int i = index; i < numValues; i += numThreads)
        sresult[threadID] += valuesIn[i];
    
    // Parallel reduction
    //for (int stride = 0 /* TODO */; stride > 0; stride /= 2) {
    for (int stride = numThreadsPerBlock / 2; stride > 0; stride /= 2) {
    
        // Synchronize threads to make sure each thread is done updating shared
        // memory; this is necessary because threads read results that have been
        // written by other threads
        __syncthreads();
        
        // Check if thread is active during this loop iteration
        //if (0 /* TODO */)
        if (threadID < stride)
        
            // Accumulate one element from sresult into another
            //sresult[threadID] += sresult[0 /* TODO */];
            sresult[threadID] += sresult[threadID + stride];
    }    
    
    // Thread 0 of each block writes the final result of the reduction to
    // device memory
    if (threadID == 0)
        valuesOut[blockID] = sresult[0];
}
