#!/bin/bash
module load cuda/12.8
module load cudnn/9.6.0.74_cuda12 miniforge3/24.11


source activate shunxian

# module load cuda/12.6 cudnn/8.9.7_cuda12.x complier/gcc/12.2.0 
# export CC=gcc
# source activate shampoo

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3
export PYTHONUNBUFFERED=1

for i in `scontrol show hostnames`
do
  let k=k+1
  let a=k-1
  host[$k]=$i
  if [ $k -eq 1 ]
  then
    torchrun \
        --nnodes=8 \
        --node_rank=0 \
        --nproc_per_node=8 \
        --master_addr="${host[1]}" \
        --master_port="3055" \
        DHO2_ADMM_v2.py >> train_rank0_"${SLURM_JOB_ID}".log 2>&1 &
  else
    srun -N 1 --gres=gpu:8 -w ${host[$k]} \
        torchrun \
        --nnodes=8 \
        --node_rank=$a \
        --nproc_per_node=8 \
        --master_addr="${host[1]}" \
        --master_port="3055" \
        DHO2_ADMM_v2.py  >> train_rank16_"${SLURM_JOB_ID}".log 2>&1 &
  fi
done
wait