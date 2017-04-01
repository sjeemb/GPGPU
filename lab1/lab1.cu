#include "lab1.h"
#include <math.h>
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 600;

__device__ static const unsigned WG = 640;
__device__ static const unsigned HG = 480;
__device__ static const unsigned NFRAMEG = 600;


struct Lab1VideoGenerator::Impl {
	int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 60;
	info.fps_d = 1;
};

__device__ static int permutation[] = { 151,160,137,91,90,15,
	   131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
	      190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
		     88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
			    77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
				   102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
				      135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
					     5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
						    223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
							   129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
							      251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
								     49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
									    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
											   };

__device__ static int p[512];


class ImprovedNoise {

public:
	__device__ static double noise(double x, double y, double z) {
		int X = (int)floor(x) & 255,
		Y = (int)floor(y) & 255,
		Z = (int)floor(z) & 255;

		x -= floor(x);
		y -= floor(y);
		z -= floor(z);
		double u = fade(x), v = fade(y), w = fade(z);
		int A = p[X]+Y, AA = p[A]+Z, AB = p[A+1]+Z,B = p[X+1]+Y, BA = p[B]+Z, BB = p[B+1]+Z;

		return lerp(w,lerp(v,lerp(u,grad(p[AA],x,y,z),grad(p[BA],x-1,y,z)),lerp(u,grad(p[AB],x,y-1,z),grad(p[BB],x-1,y-1,z))),lerp(v,lerp(u,grad(p[AA+1],x,y,z-1),grad(p[BA+1],x-1,y,z-1)),lerp(u,grad(p[AB+1],x,y-1,z-1),grad(p[BB+1],x-1,y-1,z-1))));
	}

	__device__ static double fade(double t) {return t * t * t * (t * (t * 6 - 15) + 10);}
	__device__ static double lerp(double t, double a, double b) { return a + t * (b - a);}
	__device__ static double grad(int hash, double x, double y, double z) {
		int h = hash & 15;
		double u = h < 8 ? x : y, v = h < 4 ? y : h == 12 || h == 14 ? x : z;
		return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
	}


};

__global__ void initialP(){
	for (int i=0; i < 256 ; i++) p[256+i] = p[i] = permutation[i]; 
}

__global__ void rendering(uint8_t *yuv, double tp) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= WG * HG) return;
	double wp = (double)(idx%WG);
	double hp = (double)(idx/WG); 
	yuv[idx] = (int)((255 * ( ImprovedNoise::noise(wp/45, hp/30, tp*tp*60))) *  ( 64*ImprovedNoise::noise(wp/WG*sin(tp), hp/HG*sin(tp), tp*tp*tp*5 + 5*tp)) * (ImprovedNoise::noise(wp/WG*sin(sin(tp)), hp/HG*sin(sin(tp)), tp*tp*tp*tp*5 + 5*tp)));

}

__global__ void coloring(uint8_t *yuv, double tp) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= WG * HG / 4) return;
	double wp = (double)(idx%(WG/2));
	double hp = (double)(idx/(WG/2));
	int mid_w = (int)(WG/2 * sin(sin(sin(tp))));
	int mid_h = (int)(HG/2 * ImprovedNoise::noise(wp, hp, tp*tp*tp*5+5*tp));
	yuv[idx] = 120;//(int)(255 * ImprovedNoise::noise(wp/45, hp/30, tp*tp*60));
}
void Lab1VideoGenerator::Generate(uint8_t *yuv) {
	//cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	cudaMemset(yuv, 0, W*H);
	cudaMemset(yuv+W*H, 128, W*H/2);

	initialP<<<1, 1>>>();
	cudaDeviceSynchronize();
	rendering<<<W * H / 640 + 1, 640>>>(yuv, ((double)(impl->t))/NFRAME );
	cudaDeviceSynchronize();
	coloring<<<W * H / 8 / 640 + 1, 640>>>(yuv+W*H+W*H/4, ((double)(impl->t))/NFRAME);
	cudaDeviceSynchronize();
	++(impl->t);
}

