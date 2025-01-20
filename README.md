# Pytorch + StableBaselines3 Implementation of RoboCLIP
This repository contains the implementation for the NeurIPS 2023 paper, [RoboCLIP: One Demonstration is Enough to Learn Robot Policies](https://arxiv.org/abs/2310.07899).

## Setting up the env

We recommend using conda for installation and provide a `.yml` file for installation. 

```sh
git clone https://github.com/liruiluo/RoboCLIP.git --recursive
#or git@github.com:liruiluo/RoboCLIP.git --recursive
cd RoboCLIP
wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
tar -zxvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
echo 'export LD_LIBRARY_PATH=~/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64' >> ~/.bashrc
source ~/.bashrc
sudo apt-get install -y libglew-dev
sudo apt-get update
sudo apt-get install -y build-essential gcc
conda env create -f environment_roboclip.yml
conda activate roboclip
pip install -e mjrl
pip install -e Metaworld
pip install -e kitchen_alt
pip install -e kitchen_alt/kitchen/envs
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy

#mv S3D_HowTo100M/s3dg.py ./
pip install gymnax
git clone https://github.com/carlosferrazza/humanoid-bench.git
scp install/setup.py humanoid-bench/
cd humanoid-bench/ 
pip install -e .
pip install "jax[cuda12]"
cd ..
# pip uninstall torch torchvision torchaudio
# pip uninstall jax jaxlib
# pip install torch torchvision torchaudio
conda install pytorch==2.3.1 torchvision==0.18.1 pytorch-cuda=12.1 -c pytorch -c nvidia
#pip install "jax[cuda12]"
# pip install --upgrade flax jax jaxlib
pip install stable-baselines3 --upgrade
export MUJOCO_GL=egl
```

If you're running into download issues with the S3D weights (last 2 commands), the two files can also be obtained from our google drive:
https://drive.google.com/file/d/1DN8t1feS0t5hYYTfXQfZd596Pu9yXJIr/view?usp=sharing
https://drive.google.com/file/d/1p_CMGppAQk3B0_Z0i1r2AN_D1FM3kUFW/view?usp=sharing

## How To use it ?

To run experiments on the Metaworld environment suite with the sparse learnt reward, we need to first define what the demonstration to be used is. For textual input, uncomment line 222 and comment 223 and add the string prompt you would like to use in the `text_string` param. Similarly, if you would like to use human demonstration, uncomment line 223 and pass the path of the gif of the demonstration you would like to use. Similarly, for a metaworld video demo, set `human=False` and set the `video_path`. 

We provide the gifs used in our experiments within the `gifs/`.
Then run: 
```sh
python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-run-customized-v0 --seed 0; /usr/bin/shutdown
python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-run-customized-v0 --seed 1; /usr/bin/shutdown
python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-run-customized-v0 --seed 2; /usr/bin/shutdown

python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-walk-customized-v0 --seed 0; /usr/bin/shutdown
python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-walk-customized-v0 --seed 1; /usr/bin/shutdown
python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-walk-customized-v0 --seed 2; /usr/bin/shutdown

python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-sit_hard-customized-v0 --seed 0; /usr/bin/shutdown
python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-sit_hard-customized-v0 --seed 1; /usr/bin/shutdown
python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-sit_hard-customized-v0 --seed 2; /usr/bin/shutdown

python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-stair-customized-v0 --seed 0; /usr/bin/shutdown
python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-stair-customized-v0 --seed 1; /usr/bin/shutdown
python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-stair-customized-v0 --seed 2; /usr/bin/shutdown

python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-stand-customized-v0 --seed 0; /usr/bin/shutdown
python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-stand-customized-v0 --seed 1; /usr/bin/shutdown
python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-stand-customized-v0 --seed 2; /usr/bin/shutdown

python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-balance_simple-customized-v0 --seed 0; /usr/bin/shutdown
python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-balance_simple-customized-v0 --seed 1; /usr/bin/shutdown
python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-balance_simple-customized-v0 --seed 2; /usr/bin/shutdown

python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-sit_simple-customized-v0 --seed 0; /usr/bin/shutdown
python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-sit_simple-customized-v0 --seed 1; /usr/bin/shutdown
python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-sit_simple-customized-v0 --seed 2; /usr/bin/shutdown

python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-slide-customized-v0 --seed 0; /usr/bin/shutdown
python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-slide-customized-v0 --seed 1; /usr/bin/shutdown
python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-slide-customized-v0 --seed 2; /usr/bin/shutdown

python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-balance_hard-customized-v0 --seed 0; /usr/bin/shutdown
python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-balance_hard-customized-v0 --seed 1; /usr/bin/shutdown
python humanoid_envs.py --env-type sparse_learnt --env-id h1hand-balance_hard-customized-v0 --seed 2; /usr/bin/shutdown

```

## FAQ for Debugging
Please use the older version of Metaworld, i.e., pre Farama Foundation. Also rendering can be an issue sometimes, so setting the right renderer is necessary. We found `egl` to be useful. 
```sh
export MUJOCO_GL=egl
```
