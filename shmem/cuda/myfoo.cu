#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <time.h>

//#include <common.h>
#define M 10
#define NR_BLOCK 1024

__global__ void compute(const float * a, float * b)
{
    int i = blockIdx.x;
    int j;    

    for (j = 0; j < M; j++) {
        if ((i + j * NR_BLOCK) > 0 && (i + j * NR_BLOCK) < M) {
            b[i + j * NR_BLOCK] = 0.2 * (a[M+((i+j*NR_BLOCK)-1)] + a[M+(i+j*NR_BLOCK)] + a[M+((i+j*NR_BLOCK)+1)] + a[(i+j*NR_BLOCK)] + a[2*M+(i+j*NR_BLOCK)]);
        }
    } 
}

struct params {
    float ** a;
    float ** b;
    float * c;
    float * d;
    float * c_a;
    float * c_b;
    int up, down, j;
    int stop;
    int num_pes;
    int mype;
};
typedef struct params params_t;

void foo(params_t * param)
{
    int j = param->j;
    int up = param->up;
    int down = param->down;
    int num_pes = param->num_pes;
    int mype = param->mype;
    if ((mype % 2) == 1) {
        cudaSetDevice(1);
    } else {
        cudaSetDevice(0);
    }
    // above
    if (up != -1 && j == 0) {
         cudaMemcpy(param->c_a, 
                   param->c, 
                   M * sizeof(float), 
                   cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(param->c_a, 
                   param->a[j - 1], 
                   M * sizeof(float), 
                   cudaMemcpyHostToDevice);
    } 
    // middle
    cudaMemcpy(&(param->c_a[M]), 
               param->a[j], 
               M * sizeof(float), 
               cudaMemcpyHostToDevice);

    // below
    if (down != -1 && j == param->stop - 1) {
        cudaMemcpy(&(param->c_a[2 * M]), 
                   param->d, 
                   M * sizeof(float), 
                   cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(&(param->c_a[2 * M]), 
                   param->a[j + 1], 
                   M * sizeof(float), 
                   cudaMemcpyHostToDevice);
    }
    
    cudaMemcpy(param->c_b, 
               param->b[j], 
               M * sizeof(float), 
               cudaMemcpyHostToDevice);

    compute<<<NR_BLOCK, 1>>>(param->c_a, param->c_b);
    cudaMemcpy(param->b[j], param->c_b, M * sizeof(float), cudaMemcpyDeviceToHost);
}

