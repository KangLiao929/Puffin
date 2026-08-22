#!/usr/bin/env bash

CONFIG="$1"

# Apply defaults once so unset env vars don't produce empty torchrun args.
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
WORLD_SIZE="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "WORLD_SIZE=${WORLD_SIZE}"
echo "RANK=${RANK}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"

sleep 5s

LAUNCHER="torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
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
