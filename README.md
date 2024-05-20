# Autonomous Racing with Reinforcement Learning

## Overview
This project focuses on using reinforcement learning to train AI agents for the Trackmania F-1 racing game. Utilizing the TMRL framework, we implemented deep reinforcement learning strategies to enable real-time learning and adaptation to dynamic racing conditions.

## Technical Highlights
- **Algorithm**: Soft Actor-Critic (SAC)
- **Inputs**: LIDAR data and Recurrent Neural Networks (RNNs)
- **Environments**: 
  - Pure LIDAR
  - LIDAR with track progress
  - Hybrid environments combining visual and sensory data

## Achievements
- Best lap time: 35 seconds (nearing the best human performance of 30 seconds)
- Significant improvements in training efficiency and track navigation

## Resources
- [Presentation](https://docs.google.com/presentation/d/1o_5flckV6MKzOfUubradi3Pf11HMKu0q5wjviKZX-I8/edit?usp=sharing)
- [Demo Videos](https://drive.google.com/drive/folders/1uhuof75dGtL4r3zqn8PkkF2Jo3kGyMk3?usp=drive_link)
- [Detailed Blog Post](https://ashwin2k.github.io/tmrl-rnn/)

## Introduction
Trackmania (2020) is a renowned racing game offering a blend of high-speed driving and intricate track design. Applying Reinforcement Learning (RL) to Trackmania allows for AI-driven optimization in a complex, interactive environment. Using the TMRL framework, developers can train agents to navigate the game’s challenging tracks efficiently.

## Observations and Actions
- **Observations**: Images, speed, telemetry data, and velocity norms.
- **Actions**: Analog inputs emulating an Xbox360 controller or binary arrow presses for gas, brake, steering angle, etc.

## Reward Function
Inspired by behaviorism, the environment provides rewards based on performance metrics, such as the effectiveness in covering track sections efficiently.

## Environments
1. **LIDAR Environment**: Simplified input with 19-beam LIDAR measurements, optimized for MLP models.
2. **LIDAR with Track Progress Environment**: Enhanced LIDAR environment with track completion data for predictive capabilities.
3. **Hybrid Environment**: Combines visual data and LIDAR measurements for a comprehensive training approach.

## Experiment Results
### Experiment 1: RNN Integration and Transition to LIDAR
- **Performance**: Improved training speed and track navigation, reduced collision frequency.
- **Lap Time**: 45-50 seconds

### Experiment 2: LIDAR with Track Progress Environment
- **Performance**: Rapid training progress, better anticipation of track sections.
- **Lap Time**: 35 seconds

### Experiment 3: EfficientNet_v2 in Hybrid Environment
- **Performance**: Gradual improvements but slow training, tendency to hug track edges.


# Installation
1) Clone the project
2) Navigate into the folder and do `pip install -e tmrl-drive`

## To train the model:
1) Download the Config for LIDAR from [here](https://drive.google.com/drive/u/0/folders/13rOxPTLcmqcZmrx9iUgpOW2UQQJJtYDb)
2) Use the TMRL track editor to create your desired track.
3) Next, use `python -m tmrl --record-reward` to generated the reward file. This step tracks the global points that you travel in the track, and helps the model penalize itself if it doesnt reach all the points.
4) Run these 3 commands in 3 separate terminals
   1) `python -m tmrl --server`
   1) `python -m tmrl --train`
   1) `python -m tmrl --worker`
   
   The `server` is responsible for consolidating model weights, and passing it to the trainer, while `trainer` is responsible for actually training it. `worker` is responsible for interacting with the game.

Training will take anywhere between 1-3 days on RTX 3070.

## To test the model:
1) Download the Config for LIDAR from above,
2) Paste the weights found in the folder in your home/weights folder.
3) Run `python -m tmrl --test` 

## For EffNet-V2
1) Checkout to the hybrid_environment branch:
   `git checkout hybrid_environment`
2) Follow the above steps, but use `config-imgs.json` instead of the LIDAR config.
