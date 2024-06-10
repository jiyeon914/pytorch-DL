#!/bin/bash

#SBATCH -J torch_project_test               # Job name
#SBATCH -o out.cnn_test.%j        # Name of stdout output file (%j expands to %jobId)
#SBATCH -p gpu                           # queue or partiton name
#SBATCH --gres=gpu:1                     # Num Devices

echo
python3 -V
which python3
which pip3
echo
pip3 list | grep pytorch
echo
python3 'train.py' run_name='20240610 cnn test' file_dir='/data/jykim3994/01_TurbGAN' data_dir='/data/jykim3994/0.Data/2D DHIT/resolution 128'

# End of File.

