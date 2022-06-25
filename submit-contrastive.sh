#!/bin/bash
#SBATCH --job-name=fz-openqa
#SBATCH --output=./slurm/%j.out
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH -p gpu --gres=gpu:titanrtx:4
#SBATCH --time=3-00:00:00

# variables
setup_with_model=false

# display basic info
hostname
echo $CUDA_VISIBLE_DEVICES
echo "======================"
poetry run gpustat --debug
echo "====== starting experiment ========="

# run the model
poetry run python run.py +experiment=contrastive +environ=diku \
  +patch=dpr \
  base.device_batch_size=1 \
  base.infer_batch_mul=10 \
  datamodule.dset_name=medqa-tw \
  base.eval_device_batch_size=2 \
  trainer.precision=32 \
  datamodule.num_workers=8 \
  +setup_with_model=${setup_with_model}
