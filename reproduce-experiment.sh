#!/usr/bin/env bash

curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda <<< "Yes"

export PATH="$HOME/miniconda/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/miniconda"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda install -c conda-forge mamba -y

mamba shell init --shell=bash
source ~/.bashrc  
eval "$(mamba shell hook --shell=bash)"

mamba create -n neufair python=3.9 -y
source $HOME/miniconda/bin/activate neufair
mamba activate neufair

pip install -r neufair/requirements.txt
sudo apt install -y python3-swiftclient

python main.py

source ~/openrc

bucket_name="bare_metal_experiment_pattern_data"
file_to_upload="neufair/AC-1-Runner.h5"
object_name="AC-1-Runner.h5"   

echo
echo "Uploading results to the object store container $bucket_name"

swift post "$bucket_name"

swift delete "$bucket_name" "$object_name" 2>/dev/null || true

if [ -f "$file_to_upload" ]; then
    echo "Uploading $file_to_upload"
    swift upload "$bucket_name" "$file_to_upload" --object-name "$object_name"
else
    echo "ERROR: File $file_to_upload does not exist!" >&2
    exit 1
fi