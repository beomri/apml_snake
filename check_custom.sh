#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=500M
#SBATCH --account=aml
#SBATCH --constraint="sm"

module load tensorflow
python3 Snake.py -pit 10 -D 50000 -s 5000 -pat 0.01 -plt 0.05 -P "Custom204033971();Avoid(epsilon=0.1);Avoid(epsilon=0.2);Avoid(epsilon=0.5);Avoid(epsilon=0)" 
