#!/bin/bash
#SBATCH --job-name node1gpu1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=1g
#SBATCH --time=7:00:00
#SBATCH --account=engin1
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --gpus=v100:1
#SBATCH --signal=SIGUSR1@90

conda activate MedicalNet

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# RUN PROGRAM
python3 ../train_pytorchlightning.py \
--resume_path "" \
--gpus -1 \
--num_epochs 50 \
--batch_size 16 \
--num_workers 6 \
--debug 0 \
--data_root "/home/zxbian/Datasets/train_test_data_npy_BiMask_sp1_blocksize96" \
--unet_type "4-1-4" \
--save_additional_checkpoint "" \
--cur_ckpt_loc "" \
--ID "DefaultID" \
--distributed None \
--acc_grad 1 \
--manual_seed 55 \
--precision 32 \
--optimizer "SGD" \
--optimizer_hp 0.01 0.9 1e-4 \
--scheduler "StepLR" \
--scheduler_hp 1