#!/bin/bash
#SBATCH --job-name=fz-openqa
#SBATCH --output=./slurm/%j.out
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH -p gpu --gres=gpu:titanrtx:4
#SBATCH --time=3-00:00:00

DSET_NAME=$1
CORPUS_NAME=$2
GRADIENTS=$3
PATCH=$4

echo "===================================="
echo "DSET_NAME    = ${DSET_NAME}"
echo "CORPUS_NAME  = ${CORPUS_NAME}"
echo "GRADIENTS    = ${GRADIENTS}"
echo "PATCH        = ${PATCH}"
echo "===================================="

# variables
setup_with_model=false

# set the argument for alpha
if [ "${PATCH}" != "" ]
then
     PATCH_ARG="+patch=${PATCH}"
else
     PATCH_ARG=""
fi
echo "PATCH_ARG     = ${PATCH_ARG}"


# display basic info
hostname
echo $CUDA_VISIBLE_DEVICES
echo "===================================="
poetry run gpustat --debug
echo "====== starting experiment ========="

# run the model
poetry run python run.py +experiment=option_retriever +environ=diku \
  ${PATCH_ARG} \
  model/module/gradients=${GRADIENTS} \
  datamodule.dset_name=${DSET_NAME} \
  datamodule.corpus_name=${CORPUS_NAME} \
  base.device_batch_size=2 \
  datamodule.num_workers=8 \
  +setup_with_model=${setup_with_model}
