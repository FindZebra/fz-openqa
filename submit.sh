#!/bin/bash
#SBATCH --job-name=fz-openqa
#SBATCH --output=./slurm/%j.out
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH -p gpu --gres=gpu:titanrtx:4
#SBATCH --time=7-00:00:00

# get the first arg, and shift the others (@) by one
EXP_ID="$1"
shift

# display args
echo "===================================="
echo "EXPERIMENT = $EXP_ID"
echo "ARGS       = $@"
echo "===================================="

# display basic info
hostname
echo $CUDA_VISIBLE_DEVICES
echo "===================================="
poetry run gpustat --debug

echo "====== starting experiment ========="
ES_JAVA_OPTS="-Xmx32g" HYDRA_FULL_ERROR=1 poetry run python run.py \
  +experiment=$EXP_ID \
  +environ=diku \
  datamodule.num_workers=12 \
  $@
