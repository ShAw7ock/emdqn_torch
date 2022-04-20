import os
import numpy as np
import torch as th
from pathlib import Path
from tensorboardX import SummaryWriter
from collections import deque
from core.emdqn import Agent
# from core.gemdqn import Agent
from components.arguments import parse_args
from components.replay_buffer import ReplayBuffer
from utils import wrapper


def run(env, args):
    model_dir = Path('./models') / args.experiment / args.env
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir()
                         if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)

    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    results_dir = run_dir / 'results'
    os.makedirs(str(log_dir))
    os.makedirs(str(results_dir))
    logger = SummaryWriter(str(log_dir))
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not args.use_cuda:
        th.set_num_threads(args.n_training_threads)

    # Env infos (Discrete Action Space)
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    agent = Agent(state_size, action_size, args)
    if args.load_dir is not None:
        agent.load(args.load_dir)
    buffer = ReplayBuffer(args.buffer_size)

    # Epsilon greedy
    epsilon = 0 if args.evaluate or args.layer_type == "noisy" else 1
    min_epsilon = 0.01
    anneal_steps = 3000
    anneal_epsilon = (epsilon - min_epsilon) / anneal_steps

    # Main Loop
    scores = []
    scores_window = deque(maxlen=100)

    i_episode = 0
    obs = env.reset()
    score = 0.
    for frame in range(args.n_iterations):
        if args.display:
            env.render()

        action = agent.act(obs, epsilon)
        obs_next, reward, done, info = env.step(action)

        # EMDQN
        if args.emdqn:
            agent.add_sequence(obs, action, reward)
        buffer.add(obs, action, reward, obs_next, done)
        obs = obs_next
        score += reward
        # Epsilon
        epsilon = epsilon - anneal_epsilon if epsilon > min_epsilon else epsilon

        if frame % args.learning_freq == 0 and len(buffer) > 5 * args.batch_size:
            samples = buffer.sample(args.batch_size)
            agent.train(samples)

        if done:
            # EMDQN
            if args.emdqn:
                agent.update_ec()
            scores_window.append(score)
            scores.append(score)
            print(f"\rEpisode {i_episode}\tFrame {frame} \tAverageScore: {np.mean(scores_window)}")
            logger.add_scalar("EpisodeReward", score, i_episode)
            logger.add_scalar("AverageReward", np.mean(scores_window), i_episode)
            i_episode += 1
            obs = env.reset()
            score = 0.

        if frame % args.target_update_freq == 10000:
            os.makedirs(str(run_dir / 'incremental'), exist_ok=True)
            agent.save(str(run_dir / 'incremental' / ('model_ep%i.pt' % i_episode)))
            agent.save(str(run_dir / 'model.pt'))
            if args.emdqn:
                agent.update_kdtree()

    env.close()


if __name__ == "__main__":
    args = parse_args()
    env = wrapper.make_env(args.env)
    run(env, args)

