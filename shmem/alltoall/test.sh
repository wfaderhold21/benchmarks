#!/bin/bash


# experiment 1, without hints. shows that oshmem has bound the memory to the
# NUMA associated with the PE and does not allow first touch policy to be 
# observed
orterun -np 2 \
        -hostfile ~/hostfile \
        -map-by node \
        -bind-to core \
        -x OMP_NUM_THREADS=48 \
        -x OMP_PROC_BIND=spread \
        ./alltoall-mt-far 2>&1> far.out
orterun -np 2 \
        -hostfile ~/hostfile \
        -map-by node \
        -bind-to core \
        -x OMP_NUM_THREADS=48 \
        -x OMP_PROC_BIND=spread \
        ./alltoall-mt-near 2>&1> near.out

# experiment 2, with hints. shows that our hints allow you to mbind memory 
# on different NUMA nodes as you want...
orterun -np 2 \
        -hostfile ~/hostfile \
        -map-by node \
        -bind-to core \
        -x OMP_NUM_THREADS=48 \
        -x OMP_PROC_BIND=spread \
        ./alltoall-mt-far-hint 2>&1> far-hint.out
orterun -np 2 \
        -hostfile ~/hostfile \
        -map-by node \
        -bind-to core \
        -x OMP_NUM_THREADS=48 \
        -x OMP_PROC_BIND=spread \
        ./alltoall-mt-near-hint 2>&1> near-hint.out

