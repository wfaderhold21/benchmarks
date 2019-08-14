#!/bin/bash

for j in {1..10}
do
    for i in 1 2 4 8 16 32
    do
        echo "orterun -np $i -x SMA_SYMMETRIC_SIZE=1073741824 ./5pt"
        echo "orterun -np $i --map-by socket --bind-to core -x SMA_SYMMETRIC_SIZE=1073741824 --use-hwthread-cpus ./5pt" > $i.$j.run.out
        orterun -np $i --map-by socket --bind-to core -x SMA_SYMMETRIC_SIZE=1073741824 --use-hwthread-cpus ./5pt 2>&1 >> $i.$j.run.out
    done 
done
