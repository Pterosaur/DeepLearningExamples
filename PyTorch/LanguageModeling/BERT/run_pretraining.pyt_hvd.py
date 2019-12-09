# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import csv
import os
import time
import logging
import argparse
import random
import h5py
from tqdm import tqdm, trange
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import math
import multiprocessing

from tokenization import BertTokenizer
from modeling import BertForPreTraining, BertConfig
from optimization import BertLAMB

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from utils import is_main_process
from schedulers import LinearWarmUpScheduler

import horovod.torch as hvd

from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def create_pretraining_dataset(input_file, max_pred_length, shared_list, args):

    train_data = pretraining_dataset(input_file=input_file, max_pred_length=max_pred_length)
    train_sampler = RandomSampler(train_data)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=hvd.size(), rank=hvd.rank())
    train_dataloader = DataLoader(train_data,
                                sampler=train_sampler,
                                batch_size=args.train_batch_size,
                                num_workers=4,
                                pin_memory=True)
    return train_dataloader, input_file

class pretraining_dataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, next_sentence_labels]

def parse_arguments():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .hdf5 files  for the task.")

    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")

    parser.add_argument("--bert_model", default="bert-large-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_predictions_per_seq",
                        default=80,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=1000,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=float, default=50.0,
                        help='frequency of logging loss.')
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    args = parser.parse_args()
    return args

def setup_training(args):

    assert (torch.cuda.is_available())

    # Initialize Horovod
    hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    # args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    device = torch.device("cuda", hvd.local_rank())

    return device, args


def prepare_model_and_optimizer(args, device):

    # Prepare model
    config = BertConfig.from_json_file(args.config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model = BertForPreTraining(config)

    global_step = 0

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    
    optimizer_grouped_parameters = []
    names = []

    count = 1
    for n, p in param_optimizer:
        count += 1
        if not any(nd in n for nd in no_decay):
            optimizer_grouped_parameters.append({'params': [p], 'weight_decay': 0.01, 'name': n})
            names.append({'params': [n], 'weight_decay': 0.01})
        if any(nd in n for nd in no_decay):
            optimizer_grouped_parameters.append({'params': [p], 'weight_decay': 0.00, 'name': n})
            names.append({'params': [n], 'weight_decay': 0.00})

    # optimizer = BertLAMB(optimizer_grouped_parameters,
    #                      lr=args.learning_rate,
    #                      warmup=args.warmup_proportion,
    #                      t_total=args.max_steps)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    hvd_optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    # hvd_optimizer.learning_rate = optimizer.learning_rate

    return model, hvd_optimizer, global_step


def take_optimizer_step(args, optimizer, model, global_step):
    optimizer.step()
    #optimizer.zero_grad()
    for param in model.parameters():
        param.grad = None
    global_step += 1

    return global_step

def main():

    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device, args = setup_training(args)

    # Prepare optimizer
    model, optimizer, global_step = prepare_model_and_optimizer(args, device)
    if hvd.rank() == 0:
        print("SEED {}".format(args.seed))

    if args.do_train:
        if hvd.rank() == 0:
            logger.info("***** Running training *****")
            logger.info("  Batch size = %d", args.train_batch_size)
            print("  LR = ", args.learning_rate)
            print("Training. . .")

        model.train()
        most_recent_ckpts_paths = []
        average_loss = 0.0  # averaged loss every args.log_freq steps
        epoch = 0
        training_steps = 0

        pool = ProcessPoolExecutor(1)
        # Note: We loop infinitely over epochs, termination is handled via iteration count
        while True:
            thread = None

            files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                        os.path.isfile(os.path.join(args.input_dir, f)) and 'training' in f]
            # If performance profiling, fake the training data
            if len(files) == 1:
                files = files * 1024
            files.sort()
            num_files = len(files)
            random.shuffle(files)
            f_start_id = 0


            shared_file_list = {}

            if hvd.size() > num_files:
                remainder = hvd.size() % num_files
                data_file = files[(f_start_id*hvd.size()+hvd.rank() + remainder*f_start_id)%num_files]
            else:
                data_file = files[(f_start_id*hvd.size()+hvd.rank())%num_files]

            previous_file = data_file

            train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
            # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=hvd.size(), rank=hvd.rank())
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                        batch_size=args.train_batch_size, 
                                        pin_memory=True)
            for f_id in range(f_start_id + 1 , len(files)):
                
                torch.cuda.synchronize()
                # f_start = time.time()    
                if hvd.size() > num_files:
                    data_file = files[(f_id*hvd.size()+hvd.rank() + remainder*f_id)%num_files]
                else:
                    data_file = files[(f_id*hvd.size()+hvd.rank())%num_files]

                logger.info("file no %s file %s" % (f_id, previous_file))

                previous_file = data_file

                train_iter = tqdm(train_dataloader, desc="Iteration") if hvd.rank() == 0 else train_dataloader
                for step, batch in enumerate(train_iter):
                    torch.cuda.synchronize()
                    # tracer.probe()
                    iter_start = time.time()
            
                    training_steps += 1
                    batch = [t.to(device) for t in batch]
                    
                    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
                    
                    loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                    masked_lm_labels=masked_lm_labels, next_sentence_label=next_sentence_labels)
                    loss = loss.mean()  # mean() to average on multi-gpu.

                    # t_s = time.time()
                    loss.backward()
                    # t_e = time.time()
                    # if hvd.rank() == 0:
                    #     print("backpropagation {}".format(t_e - t_s))

                    average_loss += loss.item()

                    # t_s = time.time()
                    global_step = take_optimizer_step(args, optimizer, model, global_step)
                    # t_e = time.time()
                    # if hvd.rank() == 0:
                    #     print("optimizer step {}".format(t_e - t_s))

                    if global_step >= args.max_steps:
                        last_num_steps = global_step % args.log_freq
                        last_num_steps = args.log_freq if last_num_steps == 0 else last_num_steps
                        average_loss = torch.tensor(average_loss, dtype=torch.float32).cuda()
                        average_loss = average_loss / last_num_steps
                        if (torch.distributed.is_initialized()):
                            average_loss /= torch.distributed.get_world_size()
                            torch.distributed.all_reduce(average_loss)
                        
                        logger.info("Total Steps:{} Final Loss = {}".format(training_steps, average_loss.item()))

                    if global_step >= args.max_steps:
                        del train_dataloader
                        # del tracer
                        return args


                    torch.cuda.synchronize()
                    iter_end = time.time()
                    if hvd.rank() == 0:
                        print('epoch {} global {} step {} : {}'.format(epoch, global_step, training_steps, iter_end - iter_start))
                    
                del train_dataloader

            epoch += 1


if __name__ == "__main__":
    now = time.time()
    args = main()  
    if is_main_process():
        print("Total time taken {}".format(time.time() - now))
