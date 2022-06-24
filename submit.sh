#!/bin/bash
#SBATCH --job-name=fz-openqa
#SBATCH --output=./slurm/%j.out
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH -p gpu --gres=gpu:titanrtx:4
#SBATCH --time=5-00:00:00

echo "$@"

echo "===================================="
echo "ARGS = $@"
echo "===================================="


# display basic info
hostname
echo $CUDA_VISIBLE_DEVICES
echo "===================================="
poetry run gpustat --debug

echo "====== starting experiment ========="
poetry run python run.py \
  +experiment=option_retriever \
  +environ=diku \
  datamodule.num_workers=8 \
  $@
