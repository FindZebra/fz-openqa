#!/bin/bash
#SBATCH --job-name=fz-openqa
#SBATCH --output=./slurm/%j.out
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH -p gpu --gres=gpu:titanrtx:4
#SBATCH --time=1-00:00:00

# variables
NAME="colbert-bayes-reinforce-v4.2.c-es-f15-k10-m100-diku-7"
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
    ./elasticsearch.sh &
fi

# run the model
poetry run python run.py +experiment=option_retriever +environ=diku \
  +setup_with_model=${setup_with_model} \
  datamodule.n_documents=10 \
  logger.wandb.name=${NAME} \
  model.ema_decay=null \
  model.module.use_gate=false \
  datamodule.dataset_update.freq=15 \
  trainer.gpus=4
