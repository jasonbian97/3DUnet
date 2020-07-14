#!/bin/bash
#SBATCH --job-name node1gpu1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=1g
#SBATCH --time=15:00:00
#SBATCH --account=engin1
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --gpus=v100:1

conda activate MedicalNet
python train_pytorchlightning.py --resume_path "" --gpus -1 --num_epochs 50 --milestones 30 40 --batch_size 16 --num_workers 6 --debug 0 --data_root /home/zxbian/Datasets/train_test_data_npy_BiMask_sp1_blocksize96 --unet_type "4-1-4" --save_additional_checkpoint "" --cur_ckpt_loc "" --ID "DefaultID" --amp_level "O3" --distributed None --acc_grad 1 --manual_seed 55