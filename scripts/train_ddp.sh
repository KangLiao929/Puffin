#!/usr/bin/env bash

CONFIG="$1"
echo "MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}"
echo "MASTER_PORT=${MASTER_PORT:-29500}"
echo "WORLD_SIZE=${WORLD_SIZE:-1}"
echo "RANK=${RANK:-0}"

sleep 5s

LAUNCHER="torchrun \
  --nproc_per_node=8 \
  --nnodes=${WORLD_SIZE} \
  --node_rank=${RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT}"

CMD="scripts/train.py \
  ${CONFIG} \
  --launcher pytorch \
  --deepspeed deepspeed_zero2"

echo "$LAUNCHER"
echo "$CMD"

# Run
bash -c "$LAUNCHER $CMD"
