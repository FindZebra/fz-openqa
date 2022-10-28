#!/bin/bash
#SBATCH --job-name=es-status
#SBATCH --output=./slurm/%j.out
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=100MB
#SBATCH --time=7-00:00:00

# display the host name
ES_HOST=$1
echo ${ES_HOST}
hostname

poetry run python es_status.py ${ES_HOST}
