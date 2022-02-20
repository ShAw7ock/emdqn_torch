import numpy as np
import torch as th
import gym
from core.emdqn import EMDQN
from components.arguments import parse_args
from components.replay_buffer import ReplayBuffer


def run_train(env, args):
    obs_shape = env.observation_space.shape[0]
    num_actions = env.action_space.n

    agent = EMDQN(obs_shape, num_actions, args)
    buffer = ReplayBuffer(args.buffer_size)

    # Epsilon greedy
    epsilon = 1
    min_epsilon = 0.05
    anneal_steps = 200000
    anneal_epsilon = (epsilon - min_epsilon) / anneal_steps

    # Main Training Loop
    obs = env.reset()
    for num_iter in range(args.n_iterations):
        if args.display:
            env.render()

        action = agent.select_action(obs, epsilon)
        obs_next, reward, terminate, info = env.step(action)

        # EMDQN
        agent.add_sequence(obs, action, reward)
        buffer.add(obs, action, reward, obs_next, terminate)
        obs = obs_next
        print("Iteration {} --> Reward {}".format(num_iter, reward))
        # Epsilon
        epsilon = epsilon - anneal_epsilon if epsilon > min_epsilon else epsilon

        if terminate:
            # EMDQN
            agent.update_ec()
            obs = env.reset()

        # if num_iter > max(5 * args.batch_size, args.buffer_size // 2000) and num_iter % args.learning_freq == 0:
        if num_iter > 5 * args.batch_size and num_iter % args.learning_freq == 0:
            samples = buffer.sample(args.batch_size)
            agent.learn(samples)

        if num_iter % args.target_update_freq == 100:
            agent.update_target()
            agent.update_kdtree()

    env.close()


if __name__ == "__main__":
    args = parse_args()
    env = gym.make(args.env)
    run_train(env, args)
