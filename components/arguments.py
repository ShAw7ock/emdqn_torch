import argparse


def parse_args():
    parser = argparse.ArgumentParser("DQN test for OpenAI envs")
    # Environment
    parser.add_argument("--experiment", type=str, default="emdqn", help="Name of the experiment")
    parser.add_argument("--env", type=str, required=True, help="Atari Env name")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    # DQN Parameters
    parser.add_argument("--hidden_sizes", type=int, default=512, help="hidden dimensions for MLP networks")
    parser.add_argument("--layer_type", type=str, default="ff", choices=['ff', 'noisy'])
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discounted factor")
    parser.add_argument("--tau", type=float, default=1e-3)
    parser.add_argument("--grad_norm_clip", type=float, default=10, help="gradient normalization")
    parser.add_argument("--dueling", type=str, default="dueling", help="whether dueling")
    parser.add_argument("--double_q", type=bool, default=False, help="double Q learning trick")
    parser.add_argument("--use_cuda", type=bool, default=True, help="put model/tensor to train on the CUDA")
    parser.add_argument("--display", type=bool, default=False, help="Display the rendered figures")
    parser.add_argument("--evaluate", type=bool, default=False, help="Evaluate a saved model")
    parser.add_argument("--n_training_threads", type=int, default=4, help="The training resources supplied by CPU")

    parser.add_argument("--n_iterations", type=int, default=int(400000), help="total number of training iterations")
    parser.add_argument("--batch_size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning_freq", type=int, default=4,
                        help="number of iterations between every optimization step")
    parser.add_argument("--target_update_freq", type=int, default=40000,
                        help="number of iterations between every target network update")
    # Save and Checkpoint
    parser.add_argument("--save_freq", type=int, default=10000,
                        help="save model once every time this many iterations are completed")
    parser.add_argument("--load_dir", type=str, default=None,
                        help="directory in which pre-trained model is saved")
    # EMDQN
    parser.add_argument("--emdqn", type=bool, default=True, help="EMDQN otherwise DQN")
    parser.add_argument("--ec_buffer_size", type=int, default=int(5e6), help="episodic memory size")
    parser.add_argument("--ec_latent_dim", type=int, default=4, help="dimensions for random project method")

    args = parser.parse_args()
    return args


