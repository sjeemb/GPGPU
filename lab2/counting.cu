#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct tran
{
       __host__ __device__ int operator()(const char x)
       {
              return (x=='\n') ? 0 : 1;
       }
};	

__global__ void initialize(const char* text, int *pos, int *mem int text_size) 
{
       int word=blockDim.x * blockIdx.x + threadIdx.x;
       if(word < text_size) 
       {
              pos[word]=(text[word] != '\n');
              mem[word]=pos[word];
       }
}

__global__ void countPos(const char *text, int *pos, int *mem, int text_size, int offset) 
{
       int word=blockDim.x * blockIdx.x + threadIdx.x;
       if(word < text_size) 
       {
              if(word > 0 && pos[word] && pos[word] == pos[word-1]) { 
                     mem[word] += pos[word-offset];
              }
       }
}

__global__ void setNext(int *pos, int *mem, int text_size) 
{
       int word=blockDim.x * blockIdx.x + threadIdx.x;
       if(word < text_size) 
       {
              pos[word]=mem[word];
       }
}

void CountPosition1(const char *text, int *pos, int text_size)
{
       thrust::device_ptr<const char> ptr_text(text);
       thrust::device_ptr<int> ptr_pos(pos);
       thrust::transform(ptr_text, ptr_text + text_size, ptr_pos, tran());
       thrust::inclusive_scan_by_key(thrust::device, ptr_pos, ptr_pos + text_size, ptr_pos, ptr_pos);
}

void CountPosition2(const char *text, int *pos, int text_size)
{
       int blocksize = 512;
       int num_block = CeilDiv(text_size, blocksize);
       int *mem;
       cudaMalloc(&mem, sizeof(int)*text_size);
       initialize<<<num_block, blocksize>>>(text, pos, mem, text_size);
	
       for(int i = 0; i < 10; i++) { 
              cudaDeviceSynchronize();
              countPos<<<num_block, blocksize>>>(text, pos, mem, text_size, (1<<i));
              cudaDeviceSynchronize();
              setNext<<<num_block, blocksize>>>(pos, mem, text_size);
       }
       cudaFree(mem);
}
