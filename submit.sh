#!/bin/bash
#SBATCH --job-name=fz-openqa
#SBATCH --output=./slurm/%j.out
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH -p gpu --gres=gpu:titanrtx:4
#SBATCH --time=1-00:00:00

#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES
echo "======================"
poetry run gpustat --debug
echo "====== starting experiment ========="
NAME="colbert-reinforce-v4.2.c-g.1-zero-f5-k16-diku-5"
poetry run python run.py +experiment=option_retriever +environ=diku \
  +setup_with_model=true \
  datamodule.n_documents=16 \
  datamodule.train_batch_size=4 \
  datamodule.eval_batch_size=16 \
  trainer.accumulate_grad_batches=8 \
  logger.wandb.name=${NAME} \
  model.ema_decay=null \
  model.module.use_gate=false \
  model.module.gradients.gamma=0.1 \
  datamodule.dataset_update.freq=5 \
  trainer.gpus=4
