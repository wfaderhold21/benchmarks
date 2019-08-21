#mpirun -np 8 --npernode 1 --map-by node -x UCX_TLS=dc -x UCX_NET_DEVICES=mlx5_0:1 ./alltoall-mt-near 2>&1|tee alltoall-near-nohint.8nodes.log
APPPATH="/global/home/users/manjunath/sharp-ompi-memic/benchmarks/shmem/alltoall"

APP_LIST="$APPPATH/alltoall-mt-near $APPPATH/alltoall-mt-near-hint $APPPATH/alltoall-mt-far $APPPATH/alltoall-mt-far-hint"
NODES="8"
PARAM="-np $NODES --npernode 1 --map-by node -x UCX_TLS=dc -x UCX_NET_DEVICES=mlx5_0:1"

for i_APP in `echo `$APP_LIST``
do
	CMD="mpirun $PARAM $i_APP "
	echo $CMD 
	$CMD
done
