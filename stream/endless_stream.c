#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main()
{
    const size_t size = (1ul<<29);
    uint64_t * a, *b; 
    double start, end; // timers
    double bw = 0.0;
    uint64_t iter = 0;

    a = (uint64_t *) malloc(size * sizeof(uint64_t));
    b = (uint64_t *) malloc(size * sizeof(uint64_t));


    srand(time(0));

    for (uint64_t i = 0; i < size; i++) {
        a[i] = rand();
        b[i] = 0;
    }

    printf("beginning benchmark");
    for (;;iter++) {
        start = MPI_Wtime();
        for (uint64_t i = 0; i < size; i++) {
            b[i] = a[i];
        }
        end = MPI_Wtime();
        bw = (size * sizeof(uint64_t)) / (1<<20);
        printf("[iter %lu] %10.10f\n", iter, bw / (end - start));
    }
    free(a);
    free(b);
    return 0;
}
