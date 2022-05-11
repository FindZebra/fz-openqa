#!/bin/bash
#SBATCH --job-name=fz-openqa
#SBATCH --output=./slurm/%j.out
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH -p gpu --gres=gpu:titanrtx:4
#SBATCH --time=7-00:00:00

# variables
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
poetry run python run.py +experiment=contrastive +environ=diku \
  base.device_batch_size=1 \
  base.infer_batch_mul=10 \
  base.eval_device_batch_size=2 \
  trainer.precision=32 \
  datamodule.num_workers=8 \
  +setup_with_model=${setup_with_model} \
  +kill_es=true
