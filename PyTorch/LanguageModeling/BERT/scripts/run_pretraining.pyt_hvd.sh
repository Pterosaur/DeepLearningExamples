#!/bin/bash

echo "Container nvidia build = " $NVIDIA_BUILD_ID

# export CUDA_VISIBLE_DEVICES=4,5,6,7

train_batch_size=${1:-6}
learning_rate=${2:-"6e-3"}
num_gpus=${3:-4}
train_steps=${4:-200}
create_logfile=${5:-"true"}
seed=${6:-$RANDOM}
job_name=${7:-"bert_lamb_pretraining"}

DATASET=hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus # change this for other datasets

DATA_DIR=data/${DATASET}/
BERT_CONFIG=bert_config.json
RESULTS_DIR=/results
CHECKPOINTS_DIR=/results/checkpoints

mkdir -p $CHECKPOINTS_DIR

if [ ! -d "$DATA_DIR" ] ; then
   echo "Warning! $DATA_DIR directory missing. Training cannot start"
fi
if [ ! -d "$RESULTS_DIR" ] ; then
   echo "Error! $RESULTS_DIR directory missing."
   exit -1
fi
if [ ! -d "$CHECKPOINTS_DIR" ] ; then
   echo "Warning! $CHECKPOINTS_DIR directory missing."
   echo "Checkpoints will be written to $RESULTS_DIR instead."
   CHECKPOINTS_DIR=$RESULTS_DIR
fi
if [ ! -f "$BERT_CONFIG" ] ; then
   echo "Error! BERT large configuration file not found at $BERT_CONFIG"
   exit -1
fi

echo $DATA_DIR
INPUT_DIR=$DATA_DIR
CMD=" /workspace/bert/run_pretraining.pyt_hvd.py"
CMD+=" --input_dir=$DATA_DIR"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --bert_model=bert-large-uncased"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --max_predictions_per_seq=20"
CMD+=" --max_steps=$train_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --seed=$seed"
CMD+=" --do_train"

CMD="python3 $CMD"

CMD="mpiexec --allow-run-as-root -np $num_gpus -mca btl ^openib $CMD"

if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size \* $num_gpus)
  printf -v TAG "pyt_hvd_bert_pretraining_phase1_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$job_name.$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi

