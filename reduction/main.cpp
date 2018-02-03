/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
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
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutil.h>

////////////////////////////////////////////////////////////////////////////////
// External declarations

extern "C" void reduceAllocate();
extern "C" void reduceFree();
extern "C" int* reduce(const int*, unsigned int);

////////////////////////////////////////////////////////////////////////////////
// Forward declarations

int reduceHost(const int*, unsigned int);

////////////////////////////////////////////////////////////////////////////////
//! Main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // Initialize device
    CUT_DEVICE_INIT();

    // Number of values to be added up together
    unsigned int numValues = 100000;

    // Size in memory
    unsigned int size = numValues * sizeof(int);

    // Allocate host memory for the values
    int* h_values;
    CUT_SAFE_MALLOC(h_values = (int*)malloc(size));

    // Initialize with random data
    for (unsigned int i = 0; i < numValues; ++i)
        h_values[i] = rand() & 1;

    // Allocate device memory for the values
    int* d_values;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_values, size));

    // Copy values from host memory to device memory
    CUDA_SAFE_CALL(cudaMemcpy(d_values, h_values, size, cudaMemcpyHostToDevice));

    // Create timer
    unsigned int timer;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    
    // Allocate memory used during the reduction operation
    reduceAllocate();
    
    // Device memory pointer to the sum of all values
    int* d_result;

    // Start timer to time reduction on the device
    CUT_SAFE_CALL(cutResetTimer(timer));
    CUT_SAFE_CALL(cutStartTimer(timer));

    // Loop multiple times to get accurate timing
    int numIter = 100;
    for (int i = 0; i < numIter; ++i) {
    
        // Perform reduction on the device
        d_result = reduce(d_values, numValues);

    }
    
    // Synchronize to make sure the device is done computing
    CUDA_SAFE_CALL(cudaThreadSynchronize());

    // Stop timer
    CUT_SAFE_CALL(cutStopTimer(timer));

    // Print average execution
    float time = cutGetTimerValue(timer) / numIter;
    float bandwidth = 1e-6f * (numValues * sizeof(int)) / time;
    printf("Average time is %f ms for %d values\n", time, numValues);
    printf("So average bandwidth is %f GB/s\n", bandwidth);

    // Reduce on the host
    int resultHost = reduceHost(h_values, numValues);
    
    // Retrieve result of reduction on the device
    int resultDevice;
    CUDA_SAFE_CALL(cudaMemcpy(&resultDevice, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Check host result against device result
    if (resultHost == resultDevice)
        printf("Test PASSED\n");
    else
        printf("Test FAILED (%d (device) != %d (host))\n", resultDevice, resultHost);

    // Free memory used during the reduction operation
    reduceFree();

    // Free memory
    CUDA_SAFE_CALL(cudaFree(d_values));
    free(h_values);

    // Prompt for exit
    if (argc == 1)
        CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Reduction on the host
//! @param values       input array of values to be reduced
//! @param numValues    number of values to be reduced
////////////////////////////////////////////////////////////////////////////////
int reduceHost(const int* values, unsigned int numValues)
{
    int result = 0;
    for (unsigned int i = 0; i < numValues; ++i)
        result += values[i];
    return result;
}
