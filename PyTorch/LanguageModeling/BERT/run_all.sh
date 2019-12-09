#!/bin/bash

log_dir="prof"
mkdir ${log_dir} -p

for batch_size in 1 2 3 4 5 6
do
    for gpu_count in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    do
        echo "gpu_count $gpu_count batch_size $batch_size"
        bash scripts/run_pretraining.pyt_hvd.sh $batch_size "6e-3" $gpu_count | tee ${log_dir}/gpu_count_${gpu_count}_batch_size_${batch_size}.txt
	sleep 5
    done
done
