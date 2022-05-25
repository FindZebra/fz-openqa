#!/bin/bash
#SBATCH --job-name=es-instance
#SBATCH --output=./slurm/%j.out
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=32G
#SBATCH --time=7-00:00:00

# display the host name
hostname

ES_JAVA_OPTS="-Xms16g -Xmx32g" elasticsearch
