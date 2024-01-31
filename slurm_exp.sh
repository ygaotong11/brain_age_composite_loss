#!/bin/bash
#SBATCH -p qTRDGPUL,qTRDGPUM,qTRDGPUH
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=30gb
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -e slurm_out/error%A-%a.err
#SBATCH -o slurm_out/out%A-%a.out
#SBATCH -A trends53c17
#SBATCH -J bl6
#SBATCH --oversubscribe
#SBATCH --mail-user=ygao11@gsu.edu

echo ARGS "${@:1}"

module load python
source activate jupy_cata_env

python composite_main.py --lr 1e-4 --augmentation_model LSTM_recursive --regression_model TA_LSTM --eval_mode ten_fold_cv --num_of_epoch 100 --batchsize 64 --lambda_c 0.6

