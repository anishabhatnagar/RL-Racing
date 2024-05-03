# Deep Reinceforcement Learning & Decision Making project - TrackMania Self Driving

## This projects aims to implement Self Driving AI using Reinceforment Learning In TrackMania. 

# Installation
1) Clone the project
2) Navigate into the folder and do `pip install -e tmrl-drive`

## To train the model:
1) Download the Config for LIDAR from here - https://drive.google.com/drive/u/0/folders/13rOxPTLcmqcZmrx9iUgpOW2UQQJJtYDb
2) Use the TMRL track editor to create your desired track.
3) Next, use `python -m tmrl --record-reward` to generated the reward file. This step tracks the global points that you travel in the track, and helps the model penalize itself if it doesnt reach all the points.
4) Run these 3 commands in each terminal
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
### **Checkout to hybrid_environment branch, and the follow the above steps. Note that you need to use the config-imgs.json instead!**