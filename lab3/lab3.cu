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
	if (0 <= yt && 0 <= xt && yt < ht && xt < wt && mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__global__ void PoissonImageCloningIteration(
		float *fixed, 
		const float *mask,
		float *buf1, float *buf2, // buf1 -> buf2
		int wt, int ht
		)
{
	const int yt = (blockIdx.y * blockDim.y + threadIdx.y);
	const int xt = (blockIdx.x * blockDim.x + threadIdx.x);
	const int curt = wt*yt+xt;
	if (0 <= yt && 0 <= xt && yt < ht && xt < wt ) { 
		if(mask[curt] > 127.0f) {
			int curtN = wt*(yt-1)+xt;
			int curtW = wt*yt+(xt-1);
			int curtS = wt*(yt+1)+xt;
			int curtE = wt*yt+(xt+1);
			

			for(int c = 0; c < 3; c++) {

				float sumCurrent = 0.0f;
				if(yt > 0 && mask[curtN] > 127.0f) sumCurrent += buf1[3*curtN+c];
				if(xt > 0 && mask[curtW] > 127.0f) sumCurrent += buf1[3*curtW+c];
				if(yt < ht-1 && mask[curtS] > 127.0f) sumCurrent += buf1[3*curtS+c];
				if(xt < wt-1 && mask[curtE] > 127.0f) sumCurrent += buf1[3*curtE+c];


				float pixel_around = 4.0f;
				float cb_next = (sumCurrent + fixed[curt*3+c]) / pixel_around;
				buf2[3*curt+c] = cb_next;
				
			}
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
		)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt >= 0 && xt >= 0 && yt < ht && xt < wt) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if(0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			if(mask[curt] < 127.0f) {
				// set fixed to background
				for(int c = 0; c < 3; c++) {
					fixed[3*curt+c] = background[3*curb+c];
				}
			}
			else {
				int curtN = wt*(yt-1)+xt;
				int curtW = wt*yt+(xt-1);
				int curtS = wt*(yt+1)+xt;
				int curtE = wt*yt+(xt+1);

				int curbN = wb*(yb-1)+xb;
				int curbW = wb*yb+(xb-1);
				int curbS = wb*(yb+1)+xb;
				int curbE = wb*yb+(xb+1);

				for(int c = 0; c < 3; c++) {
					float tgradient = 0.0f;
					if(yt > 0) tgradient += target[3*curt+c] - target[3*curtN+c];
					if(xt > 0) tgradient += target[3*curt+c] - target[3*curtW+c];
					if(yt < ht-1) tgradient += target[3*curt+c] - target[3*curtS+c];
					if(xt < wt-1) tgradient += target[3*curt+c] - target[3*curtE+c];

					float vborder = 0.0f;
					if(yt == 0 || mask[curtN] < 127.0f) vborder += background[3*curbN+c];
					if(xt == 0 || mask[curtW] < 127.0f) vborder += background[3*curbW+c];
					if(yt == ht-1 || mask[curtS] < 127.0f) vborder += background[3*curbS+c];
					if(xt == wt-1 || mask[curtE] < 127.0f) vborder += background[3*curbE+c];

					fixed[3*curt+c] = tgradient + vborder;

				}
			}
		}

	}
}

__global__ void scaling(
		const float *mask,
		float *fixed,
		float *buf1,
		float *mask_scaled,
		float *fixed_scaled,
		float *buf1_scaled,
		const int wt, const int ht,
		int scale
		)
{
	const int wt_scaled = wt/scale;
	const int ht_scaled = ht/scale;
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt_scaled*yt+xt;
	const int curb = wt*yt*scale+xt*scale;
	if(yt >= 0 && xt >= 0 && yt < ht_scaled && xt < wt_scaled) {
		for(int c = 0; c < 3; c++) {
			float sumf = 0.0f, sumb = 0.0f;
			float countf = 0.0f, countb = 0.0f;
			for(int i = xt*scale; i < xt*scale+scale && i < wt; i++) {
				for(int j = yt*scale; j < yt*scale+scale && j < ht; j++) {
					sumf += fixed[3*(j*wt+i)+c];
					sumb += buf1[3*(j*wt+i)+c];
					countf += 1.0;
					countb += 1.0;
				}
			}
			sumf /= countf;
			sumb /= countb;
			fixed_scaled[3*curt+c] = sumf;
			buf1_scaled[3*curt+c] = sumb;
		}
		float summ = 0.0f;
		float countm = 0.0f;

		for(int i = xt*scale; i < xt*scale+scale && i < wt; i++) {
			for(int j = yt*scale; j < yt*scale+scale && j < ht; j++) {
				summ += mask[(j*wt+i)];
				countm += 1.0;
			}
		}
		summ /= countm;

		mask_scaled[curt] = summ;

	}
}

__global__ void rescaling(
		float *buf1,
		float *buf1_scaled,
		const int wt, const int ht,
		int scale
		)
{
	
	const int wt_scaled = wt/scale;
	const int ht_scaled = ht/scale;
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt_scaled*yt+xt;
	if(yt >= 0 && xt >= 0 && yt < ht_scaled && xt < wt_scaled) {
		for(int i = xt*scale; i < xt*scale+scale && i < wt; i++) {
			for(int j = yt*scale; j < yt*scale+scale && j < ht; j++) {
				for(int c = 0; c < 3; c++) {
					buf1[3*(j*wt+i)+c] = buf1_scaled[3*curt+c];
				}
			}
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

	float *fixed_scaled, *buf1_scaled, *mask_scaled;
	cudaMalloc(&fixed_scaled, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1_scaled, 3*wt*ht*sizeof(float));
	cudaMalloc(&mask_scaled, wt*ht*sizeof(float));
	
	dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32, 16);

	CalculateFixed<<<gdim, bdim>>>(
			background, target, mask, fixed,
			wb, hb, wt, ht, oy, ox
			);
	cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);

	int iterations = 2000;
	for(int i = 0; i < iterations; i++) {
		int scale = (i<100) ? 16 : (i<200) ? 8 : (i<300) ? 4 : (i<400) ? 2 : 1;
		
		dim3 gdim(CeilDiv(wt, 32*scale), CeilDiv(ht, 16*scale)), bdim(32, 16);
		if(i==0 || i == 100 || i == 200 || i == 300 || i == 400) {
			scaling<<<gdim, bdim>>>(
					mask, fixed,  buf1, mask_scaled, fixed_scaled, buf1_scaled, wt, ht, scale
					);
		}
		PoissonImageCloningIteration<<<gdim, bdim>>>(
				fixed_scaled, mask_scaled, buf1_scaled, buf2, wt/scale, ht/scale
				);
		PoissonImageCloningIteration<<<gdim, bdim>>>(
				fixed_scaled, mask_scaled, buf2, buf1_scaled, wt/scale, ht/scale
				);
		if(i == 100-1 || i == 200-1 || i == 300-1 || i == 400-1 ) {
			rescaling<<<gdim, bdim>>>(
						buf1, buf1_scaled, wt, ht, scale
						);
		}
	}

	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);

	SimpleClone<<<gdim, bdim>>>(
			background, buf1_scaled, mask, output,
			wb, hb, wt, ht, oy, ox
			);

	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);

}

