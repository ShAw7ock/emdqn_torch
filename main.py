import os
import numpy as np
import torch as th
import gym
from pathlib import Path
from core.emdqn import EMDQN
from components.arguments import parse_args
from components.replay_buffer import ReplayBuffer


def run_train(env, args):
    model_dir = Path('./models') / args.env
    algo_name = "emdqn" if args.emdqn else "dqn"
    if not model_dir.exists():
        curr_run = "run1"
    else:
        exist_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir()
                          if str(folder.name).startswith('run')]
        if len(exist_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exist_run_nums) + 1)

    run_dir = model_dir / algo_name / curr_run
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not args.use_cuda:
        th.set_num_threads(args.n_training_threads)

    # Env infos (Discrete Action Space)
    obs_shape = env.observation_space.shape[0]
    num_actions = env.action_space.n

    agent = EMDQN(obs_shape, num_actions, args)
    if args.load_dir is not None:
        agent.load(args.load_dir)
    buffer = ReplayBuffer(args.buffer_size)

    # Epsilon greedy
    epsilon = 0 if args.evaluate and args.load_dir is not None else 1
    min_epsilon = 0.05
    anneal_steps = 3000
    anneal_epsilon = (epsilon - min_epsilon) / anneal_steps

    # Main Training Loop
    obs = env.reset()
    iter_rew = 0.
    for num_iter in range(args.n_iterations):
        if args.display:
            env.render()

        action = agent.select_action(obs, epsilon)
        obs_next, reward, terminate, info = env.step(action)
        # Refine the Reward for MountainCar-v0: the higher, the better
        position, velocity = obs_next
        reward = abs(position - (-0.5))
        iter_rew += reward

        # EMDQN
        if args.emdqn:
            agent.add_sequence(obs, action, reward)
        buffer.add(obs, action, reward, obs_next, terminate)
        obs = obs_next
        # Epsilon
        epsilon = epsilon - anneal_epsilon if epsilon > min_epsilon else epsilon

        if terminate:
            # EMDQN
            if args.emdqn:
                agent.update_ec()
            obs = env.reset()
            print(f"Iteration: {num_iter} |-> Get! |-> Reward: {iter_rew}")
            iter_rew = 0.

        # if num_iter > max(5 * args.batch_size, args.buffer_size // 2000) and num_iter % args.learning_freq == 0:
        if not args.evaluate and num_iter > 5 * args.batch_size and num_iter % args.learning_freq == 0:
            samples = buffer.sample(args.batch_size)
            agent.learn(samples)

        if not args.evaluate and num_iter % args.target_update_freq == 100:
            agent.update_target()
            # EMDQN
            if args.emdqn:
                agent.update_kdtree()

        if not args.evaluate and num_iter % args.save_freq == 0:
            os.makedirs(str(run_dir / 'incremental'), exist_ok=True)
            agent.save(str(run_dir / 'incremental' / ('model_iter%i.pt' % (num_iter + 1))))
            agent.save(str(run_dir / 'model.pt'))

    agent.save(str(run_dir / 'model.pt'))
    env.close()


if __name__ == "__main__":
    args = parse_args()
    env = gym.make(args.env)
    env = env.unwrapped
    run_train(env, args)
