#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=2-00:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=george.stamatescu@student.adelaide.edu.au

source $HOME/.bashrc

pyenv activate pytorch

export DATA_DIR=/fast/users/a1195560/NNcorrelations
export CHECKPOINT_DIR=/fast/users/a1195560/corrChkpts

python cuda_allClass_L2.py 
-num_views=2 --use_context --teacher_forcing=1.0 --vgg_loss
