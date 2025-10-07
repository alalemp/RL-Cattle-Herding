# gym-pybullet-drones

This repo is based off of the gym-pubullet-drones repo which can be found here:
https://github.com/utiasDSL/gym-pybullet-drones

<img src="gym_pybullet_drones/assets/helix.gif" alt="formation flight" width="325"> <img src="gym_pybullet_drones/assets/helix.png" alt="control info" width="425">

## Installation

Developed for x64/Ubuntu 22.04

```sh
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones/

conda create -n drones python=3.10
conda activate drones

pip3 install --upgrade pip
pip3 install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`

```

## Development /Working Files
envs -> AdvancedCurriculumnAviary - enhanced learning environment with curriculumn learning
        BaseAviary - handles bulk of simulation 
        BaseRLAvairy - handles action and opbservation spaces
        CattleAviary - computes reward, termination, truncated conditions
models -> saved trained models
simulator -> CattleHerder - main file for running cattle herding

utils -> flockUtils - 
         mathUtils -
         utils


## Cattle Herding Training

```sh
cd gym_pybullet_drones/simulator/
python CattleHerder.py --num_drones 6 --num_cattle 16 --gui False
```

## Cattle Herding Evaluation

```sh
cd gym_pybullet_drones/simulator/
python CattleHerder.py --num_drones 6 --num_cattle 16 --gui True --eval_only True --eval_episode_length 1
```

<img src="gym_pybullet_drones/assets/rl.gif" alt="rl example" width="375"> <img src="gym_pybullet_drones/assets/marl.gif" alt="marl example" width="375">
