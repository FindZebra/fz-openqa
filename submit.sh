#!/bin/bash
#SBATCH --job-name=fz-openqa
#SBATCH --output=./slurm/%j.out
#SBATCH --ntasks=1 --cpus-per-task=32 --mem=128G
#SBATCH -p gpu --gres=gpu:titanrtx:4
#SBATCH --time=1-00:00:00

# variables
NAME="xyt-DIKU-lscaled-xmatch-inbatch-v4.4.A-es-f15-k10-P1000-9.0"
setup_with_model=false

# display basic info
hostname
echo $CUDA_VISIBLE_DEVICES
echo "======================"
poetry run gpustat --debug
echo "====== starting experiment ========="

# startup elastic search
if [ "$setup_with_model" = false ]
then
    elasticsearch --quiet &
fi

# run the model
poetry run python run.py +experiment=option_retriever +environ=diku \
  model/module/gradients=in_batch \
  base.device_batch_size=2 \
  trainer.precision=16 \
  datamodule.num_workers=4 \
  +setup_with_model=${setup_with_model} \
  +kill_es=true \
  logger.wandb.name=${NAME}
