Episodic Memory DQN by PyTorch
================
## EMDQN
This repo is a simple test code for ["Episodic Memory Deep Q-Networks"](https://arxiv.org/pdf/1805.07603.pdf) 
by Zichuan Lin, Tianqi Zhao, Guangwen Yang and Lintao Zhang.
And their EMDQN source code can be gotten [here](https://github.com/LinZichuan/emdqn) using `Tensorflow`.<br>

This code is a simple version using `PyTorch`.<br>
The users can modify the code to suit your own testing environments (`DISCREATE ACTION SPACE`).

<p align="center">
 Episodic Memory DQN<br>
 <img src="https://github.com/LinZichuan/emdqn/blob/master/data/emdqn.PNG" width="352" height="352"><br>
</p>

## Requirements:
* Python >= 3.6.0 (optional)
* PyTorch == 1.7.0 (optional)
* OpenAI Gym[Atari]
* Scikit-Learn == 1.0.2 (optional)

## NOTE:
* To run this code, `cd` into the root directory and run : `python main.py --env PongNoFrameskip-v4`
* The kernel updating codes for EMDQN algorithm: `./core/emdqn.py`
* The Episodic Memory using LRU_KNN: `./utils/lru_knn.py`
* The off-policy replay buffer: `./components/replay_buffer.py`
* The networks include base Q network and Dueling Q network: `./components/networks.py`
* The Hyper-parameters can be modified in: `./components/arguments.py`
The details can be seen in the original [paper](https://arxiv.org/pdf/1805.07603.pdf).

## TODO LIST:
- [x] DQN, DuelingDQN, EMDQN
- [x] Save and Load Pre-Trained Models.
- [ ] CUDA supported.
- [x] Modify and Suit Atari Games.
- [ ] Modify Prioritized ReplayBuffer.
