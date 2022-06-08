#!/bin/bash
#SBATCH --job-name=fz-openqa
#SBATCH --output=./slurm/%j.out
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH -p gpu --gres=gpu:titanrtx:4
#SBATCH --time=3-00:00:00

DSET_NAME=$1
CORPUS_NAME=$2
GRADIENTS=$3
TEMPERATURE=$4
ALPHA=""

echo "===================================="
echo "DSET_NAME    = ${DSET_NAME}"
echo "CORPUS_NAME  = ${CORPUS_NAME}"
echo "GRADIENTS    = ${GRADIENTS}"
echo "ALPHA        = ${ALPHA}"
echo "TEMPERATURE  = ${TEMPERATURE}"
echo "===================================="

# variables
setup_with_model=false

# set the argument for alpha
if [ "${ALPHA}" != "" ]
then
     ALPHA_ARG="model.parameters.alpha=${ALPHA}"
else
     ALPHA_ARG=""
fi
echo "ALPHA_ARG     = ${ALPHA_ARG}"
if [ "${TEMPERATURE}" != "" ]
then
     TEMPERATURE_ARG="datamodule.index_builder.engines.es.config.es_temperature=${TEMPERATURE}"
else
     TEMPERATURE_ARG=""
fi
echo "TEMPERATURE_ARG= ${TEMPERATURE_ARG}"


# display basic info
hostname
echo $CUDA_VISIBLE_DEVICES
echo "===================================="
poetry run gpustat --debug
echo "====== starting experiment ========="

# run the model
poetry run python run.py +experiment=option_retriever +environ=diku \
  +patch=dpr \
  model/module/gradients=${GRADIENTS} \
  datamodule.dset_name=${DSET_NAME} \
  datamodule.corpus_name=${CORPUS_NAME} \
  trainer.precision=32 \
  datamodule.num_workers=8 \
  +setup_with_model=${setup_with_model} \
  ${ALPHA_ARG} \
  ${TEMPERATURE_ARG}
