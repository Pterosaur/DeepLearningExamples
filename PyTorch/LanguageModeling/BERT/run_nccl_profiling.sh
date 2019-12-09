#!/bin/bash

pwd=$(cd $(dirname ${0}) && pwd)
log_dir=${pwd}/"nccl-logs"
nccl_test_dir=${pwd}/"nccl-tests"
gpu_total_count=$(ls -l /dev | grep -E "nvidia[0-9]" | wc -l)

mkdir -p ${log_dir}

if [ ! -d ${nccl_test_dir} ]
then
    git clone https://github.com/NVIDIA/nccl-tests.git ${nccl_test_dir}
fi

cd ${nccl_test_dir}
make
for op in "all_gather" "all_reduce" "broadcast" "reduce" "reduce_scatter"
do
    for gpu_count in 1 2 4 8 16
    do
        if [ "${gpu_count}" -le "${gpu_total_count}" ]
        then
            echo "gpu_count ${op} op ${op}"
            ${nccl_test_dir}/build/${op}_perf -b 8M -e 1G -f 2 -g ${gpu_count} | tee ${log_dir}/gpu_count_${gpu_count}_op_${op}.txt
        fi
    done
done
