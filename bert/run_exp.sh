#!/usr/bin/bash

#SBATCH --cpus-per-task 2
#SBATCH --mem-per-cpu=24GB
#SBATCH --gres=gpu:v100:2
#SBATCH --job-name=bert_aeda
#SBATCH --output ./log/%J-%x.log
#SBATCH --time 2-00:00:00

export CUDA_VISIBLE_DEVICES=0,1

source activate llm

python bert/aedberta2.py