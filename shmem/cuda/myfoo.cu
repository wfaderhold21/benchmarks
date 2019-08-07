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
};
typedef struct params params_t;

void foo(params_t * param)
{
    int j = param->j;
    int up = param->up;
    int down = param->down;
    int num_pes = param->num_pes;
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
    /*if (!(down != -1 && j == (M / num_pes - 2))) {
        cudaMemcpy(&(param->c_a[2 * M]), 
                   param->a[j + 1], 
                   M * sizeof(float), 
                   cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(&(param->c_a[2 * M]), 
                   param->d, 
                   M * sizeof(float), 
                   cudaMemcpyHostToDevice);
    }*/
    cudaMemcpy(param->c_b, 
               param->b[j], 
               M * sizeof(float), 
               cudaMemcpyHostToDevice);

    compute<<<NR_BLOCK, 1>>>(param->c_a, param->c_b);
    cudaMemcpy(param->b[j], param->c_b, M * sizeof(float), cudaMemcpyDeviceToHost);
}
           /*cudaMemcpy(c_a, a[j - 1], M * sizeof(float), cudaMemcpyHostToDevice);
           cudaMemcpy(&c_a[M], a[j], M * sizeof(float), cudaMemcpyHostToDevice);
           cudaMemcpy(&c_a[2*M], a[j+1], M * sizeof(float), cudaMemcpyHostToDevice);
           
           cudaMemcpy(b[j], c_b, M * sizeof(float), cudaMemcpyDeviceToHost); */
       //}
       
//       printf("[debug] updating a with b\n");
/*       for (j = 1; j < M - 1; j++) {
           for (k = 1; k < M - 1; k++) {
               a[j][k] = b[j][k];
           }
       }*/
/*       #ifdef DEBUG
       printf("[debug output of b]\n");
       for (j = 0; j < M; j++) {
           for (k = 0; k < M; k++) {
               printf("%5.5g ", a[j][k]);
           }
           printf("\n");
       }
       printf("\n\n");
       #endif * DEBUG * */
    //}   
//}
