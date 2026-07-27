#!/bin/bash
cd "$(dirname "$0")/.."   # run from the repo root
# Single-node example. Adjust GPUs / paths for your setup.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_PROCS_PER_NODE=4

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
export NODE_RANK=0
export SLURM_NNODES=1

torchrun \
  --nnodes=$SLURM_NNODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --nproc_per_node=$N_PROCS_PER_NODE \
  --master_port=$MASTER_PORT \
  -m runners.run_lens \
  --itm_model_name CLIP \
  --model_name qwenvl25_7b \
  --task videomme \
  --data_path /path/to/Video-MME \
  --num_frames 32
