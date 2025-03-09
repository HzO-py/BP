#!/bin/bash -l 
#SBATCH -J lzqhzo_ppg2abp_pytorch
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=15-00:00:00
#SBATCH --mail-user=zqliang@ucdavis.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -o bench-%j.output
#SBATCH -e bench-%j.output
#SBATCH --partition=gpu-homayoun
#SBATCH --gres=gpu:1
hostname
# python data_handling.py
python -u train_models.py
python -u predict_test.py
# python evaluate.py