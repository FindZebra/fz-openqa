#!/bin/bash
#SBATCH --job-name=fz-openqa
#SBATCH --output=./slurm/%j.out
#SBATCH --ntasks=1 --cpus-per-task=24 --mem=128G
#SBATCH -p gpu --gres=gpu:titanrtx:4
#SBATCH --time=1-00:00:00

# variables
NAME="xyt-DIKU-fxmatch-inbatch-v4.4.A-L350-9.3.1-attn-rop0.5"
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
  base.sharing_strategy=file_descriptor \
  trainer.precision=32 \
  datamodule.num_workers=12 \
  datamodule.builder.dataset_builder.max_length=350 \
  model.bert.config.attention_probs_dropout_prob=0.5 \
  +setup_with_model=${setup_with_model} \
  +kill_es=true \
  logger.wandb.name=${NAME}
