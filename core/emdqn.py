import numpy as np
import torch as th
from components.networks import QNetwork
from utils.misc import hard_update, soft_update
from utils.lru_knn import LRUKnn


class Agent:
    def __init__(self, state_size, action_size, args):
        self.state_size = state_size
        self.action_size = action_size

        self.qnet_eval = QNetwork(state_size, action_size,
                                  hidden_sizes=args.hidden_sizes, layer_type=args.layer_type)
        self.qnet_target = QNetwork(state_size, action_size,
                                    hidden_sizes=args.hidden_sizes, layer_type=args.layer_type)

        if args.use_cuda and th.cuda.is_available():
            self.device = th.device("cuda:0")
            self.qnet_eval.to(self.device)
            self.qnet_target.to(self.device)
        else:
            self.device = th.device("cpu")

        hard_update(self.qnet_target, self.qnet_eval)
        self.parameters = list(self.qnet_eval.parameters())
        self.optimizer = th.optim.Adam(self.parameters, lr=args.lr)

        self.args = args

        # EMDQN
        if args.emdqn:
            self.ec_buffer = []
            for i in range(self.action_size):
                self.ec_buffer.append(LRUKnn(args.ec_buffer_size, args.ec_latent_dim, env_name=args.env))
            # random project method
            rng = np.random.RandomState(123456)
            self.rp = rng.normal(loc=0, scale=1./np.sqrt(args.ec_latent_dim), size=(args.ec_latent_dim, 84 * 84 * 4))
            self.qec_watch = []
            self.update_counter = 0
            self.qec_found = 0
            self.sequence = []

    def act(self, state, epsilon=0.):
        if np.random.uniform() < epsilon:
            action = np.random.choice(self.action_size)
        else:
            state = np.array(state)
            state = th.tensor(state, dtype=th.float32).unsqueeze(0).to(self.device)
            with th.no_grad():
                action_values = self.qnet_eval(state)
            action = np.argmax(action_values.cpu().data.numpy())

        return action

    def add_sequence(self, state, action, reward):
        self.sequence.append([np.array(state), action, reward])

    def update_ec(self):
        mt_rew = 0.
        for seq in reversed(self.sequence):
            o, u, r = seq
            z = np.dot(self.rp, o.flatten())
            mt_rew = r + self.args.gamma * mt_rew
            z = z.reshape(self.args.ec_latent_dim)
            qd = self.ec_buffer[u].peek(z, mt_rew, modify=True)
            if qd is None:
                self.ec_buffer[u].add(z, mt_rew)
        # Clear
        self.sequence = []

    def update_kdtree(self):
        for u in range(self.action_size):
            self.ec_buffer[u].update_kdtree()

    def train(self, batch):
        if self.args.emdqn:
            qec_inputs = self._get_qec_inputs(batch)

        for key in batch.keys():
            if key == 'u':
                batch[key] = th.tensor(batch[key], dtype=th.long).to(self.device)
            else:
                batch[key] = th.tensor(batch[key], dtype=th.float32).to(self.device)
        o, u, o_next = batch['o'], batch['u'], batch['o_next']
        r, terminates = batch['r'], batch['terminates']

        q_values = self.qnet_eval(o)
        q_targets = self.qnet_target(o_next)

        q_eval = th.gather(q_values, dim=1, index=u).squeeze(1)
        if self.args.double_q:
            q_targets_eval = self.qnet_eval(o_next)
            trgt_eval_u = th.max(q_targets_eval, dim=1, keepdim=True)[1]
            q_trgt = th.gather(q_targets, dim=1, index=trgt_eval_u).squeeze(1)
        else:
            q_trgt = th.max(q_targets, dim=1)[0]
        target = r + self.args.gamma * (1 - terminates) * q_trgt

        td_error = target.detach() - q_eval
        if self.args.emdqn:
            qec_error = qec_inputs - q_eval
            loss = td_error.pow(2).mean() + 0.1 * qec_error.pow(2).mean()
        else:
            loss = td_error.pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        soft_update(self.qnet_target, self.qnet_eval, self.args.tau)

    def _get_qec_inputs(self, batch):
        self.update_counter += 1
        o_array, u_array = batch['o'], batch['u']
        o_tensor = th.tensor(o_array, dtype=th.float32).to(self.device)
        u_tensor = th.tensor(u_array, dtype=th.long).to(self.device)

        qec_values = self.qnet_eval(o_tensor)
        qec_selected = th.gather(qec_values, dim=1, index=u_tensor).squeeze(1)
        # transfer to array for ec_buffer
        qec_selected = qec_selected.cpu().data.numpy().reshape(self.args.batch_size)
        # u_array[i] can then an index
        u_array = np.squeeze(u_array, axis=1)
        for i in range(self.args.batch_size):
            z = np.dot(self.rp, o_array[i].flatten())
            q = self.ec_buffer[u_array[i]].peek(z, None, modify=False)
            if q is not None:
                qec_selected[i] = q
                self.qec_watch.append(q)
                self.qec_found += 1

        if self.update_counter % 2000 == 1999:
            print(f"qec_mean: {np.mean(self.qec_watch)}")
            print("qec_fount: %.2f" % (1.0 * self.qec_found / self.args.batch_size / self.update_counter))
            # Clear
            self.qec_watch = []
        qec_inputs = th.tensor(qec_selected, dtype=th.float32).to(self.device)

        return qec_inputs   # shape: [batch_size]

    def save(self, filename):
        param_dict = {
            "qnet_eval": self.qnet_eval.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        th.save(param_dict, filename)

    def load(self, filename):
        params_dict = th.load(filename)
        # Get parameters from save_dict
        self.qnet_eval.load_state_dict(params_dict["qnet_eval"])
        self.optimizer.load_state_dict(params_dict["optimizer"])
        # Copy the eval networks to target networks
        hard_update(self.qnet_target, self.qnet_eval)
