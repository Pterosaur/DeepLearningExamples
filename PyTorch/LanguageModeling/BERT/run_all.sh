#!/bin/bash

pwd=$(cd $(dirname ${0}) && pwd)
log_dir=${pwd}/"bert-logs"
gpu_total_count=$(ls -l /dev | grep -E "nvidia[0-9]" | wc -l)

mkdir ${log_dir} -p

for batch_size in 1 2 3 4 5 6
do
    for gpu_count in 1 2 4 8 16
    do
        if [ "${gpu_count}" -le "${gpu_total_count}" ]
        then
            echo "gpu_count $gpu_count batch_size $batch_size"
            bash ${pwd}/scripts/run_pretraining.pyt_hvd.sh $batch_size "6e-3" $gpu_count | tee ${log_dir}/gpu_count_${gpu_count}_batch_size_${batch_size}.txt
	        sleep 5
        fi
    done
done
