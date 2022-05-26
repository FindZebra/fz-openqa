#!/bin/bash
#SBATCH --job-name=fz-race
#SBATCH --output=./slurm/%j.out
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH -p gpu --gres=gpu:titanrtx:2
#SBATCH --time=7-00:00:00

# variables
NAME="xytx-DIKU-reiA-colbert-cosine"

# display basic info
hostname
echo $CUDA_VISIBLE_DEVICES
echo "======================"
poetry run gpustat --debug
echo "====== starting experiment ========="

# run the model
poetry run python run.py \
  +experiment=option_retriever +environ=diku +patch=race \
  base.device_batch_size=4 \
  base.eval_device_batch_size=8 \
  base.infer_batch_mul=100 \
  trainer.precision=32 \
  datamodule.num_workers=8 \
  +spawn_es=true \
  logger.wandb.name=${NAME}
