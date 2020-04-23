#!/bin/bash

#SBATCH --job-name=yg390
#SBATCH --mail-user=yg390@nyu.edu
#SBATCH --mail-type=END
#SBATCH --mem=12GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH -o fastmri_test.log
#SBATCH --time=5:00:00

# activate existed python3 module to get virtualenv
module purge
#module load python3/intel/3.6.3
#module load pytorch/python3.6/0.3.0_4
#module load cudnn/10.1v7.6.5.32
#module load cuda/10.0.130


module load python3/intel/3.6.3

# activate virtual environment
source ~/fastmri/bin/activate

export PYTHONPATH="/beegfs/yg390/fastMRI"
cd /beegfs/yg390/fastMRI
python models/neumann/train_neumann.py --batch_size 3 --gpu 1 --mode test --num-epoch 1 --exp-dir experiments/neumann --exp 20200421_test_neumann --challenge singlecoil --data-path data --checkpoint experiments/20200420_neumann/version_6/checkpoints/epoch=49.ckpt
