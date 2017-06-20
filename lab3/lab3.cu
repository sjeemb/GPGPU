#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}


__global__ void CalculateFixed(
	const float *background,
	const float *target,
	const float *mask,
	float *fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
){

	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	const int yb=oy+yt, xb=ox+xt;
	int nt=wt*(yt-1)+xt,nb=wb*(yb-1)+xb,st=wt*(yt+1)+xt,sb=wb*(yb+1)+xb;
	int et=wt*yt+xt+1,eb=wb*yb+xb+1;
	int wtt=wt*yt+xt-1,wtb=wb*yb+xb-1;
	int dir[8]={nt*3,wtt*3,et*3,st*3,nb*3,wtb*3,eb*3,sb*3};
	bool condi[8]={yt-1 >= 0,
		       xt-1 >= 0,
                       xt+1 < wt,
                       yt+1 < ht,
                       yt-1 >=0 && !mask[nt],
                       xt-1 >=0 && !mask[wtt],
                       xt+1 < wt && !mask[et],
                       yt+1 < ht && !mask[st]};
	if(yt<ht && xt<wt){
		for(int k=0;k<3;k++){
			fixed[curt*3+k]=4*target[curt*3+k];
		}
		for(int l=0;l<4;l++){
			if(condi[l]){
				for(int m=0;m<3;m++){
					fixed[curt*3+m]=fixed[curt*3+m]-target[dir[l]+m];
				}
			}
		}
		if(0<=yb && yb<hb && 0<=xb && xb<wb){
			for(int i=4;i<8;i++){
				if(condi[i]){
					for(int j=0;j<3;j++){
						fixed[curt*3+j]=fixed[curt*3+j]+ background[dir[i]+j];
					}
				}
				else if(!condi[i-4]){
					for(int j=0;j<3;j++){
						fixed[curt*3+j]=background[dir[i]+j];
					}
						
				}
			}		
		}
	}

}


__global__ void PoissonImageCloningIteration(
	float *fixed,
	const float *mask,	
	float *input,
	float *output,
	const int wt, const int ht
){

	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	
	if (yt < ht && xt < wt && mask[curt] > 127.0f) {
		for(int i=0;i<3;i++){
			output[curt*3+i]=0;
		}
		if( xt-1>=0 && mask[curt-1] > 127.0f){    
			for(int i=0;i<3;i++){
			output[curt*3+i] = output[curt*3+i]+input[(curt-1)*3+i]*1/4;
			}
		}
		if( xt+1<wt && mask[curt+1] > 127.0f){
			for(int i=0;i<3;i++){
			output[curt*3+i] = output[curt*3+i]+ input[(curt+1)*3+i]*1/4;
			}
		}
		if( yt-1>=0 && mask[curt-wt] > 127.0f ){
			for(int i=0;i<3;i++){
			output[curt*3+i]=output[curt*3+i]+input[(curt-wt)*3+i]*1/4;
			}   
		}
		if( yt+1<ht && mask[curt+wt] > 127.0f){   
			for(int i=0;i<3;i++){
			output[curt*3+i]=output[curt*3+i]+input[(curt+wt)*3+i]*1/4;
			}	
		 }

		for(int i=0;i<3;i++){
			output[curt*3+i] = output[curt*3+i]+fixed[curt*3+i]*1/4;
		}
	}	
}


void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

	dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
	CalculateFixed<<<gdim, bdim>>>(
		background, target, mask, fixed,
		wb, hb, wt, ht, oy, ox
    	);
    	cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);
	
    	for (int i = 0; i < 20000; ++i) {
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf1, buf2, wt, ht
		);
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf2, buf1, wt, ht
		);
    	}
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    	SimpleClone<<<gdim, bdim>>>(
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
	);
	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
}
