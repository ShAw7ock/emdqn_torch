import torch as th
import numpy as np
from utils.networks import MLPNetwork
from utils.lru_knn import LRUKnn


class EMDQN:
    def __init__(self, obs_dim, num_actions, args):
        self.obs_dim = obs_dim
        self.n_actions = num_actions

        self.eval_mlp = MLPNetwork(obs_dim, num_actions, hidden_dims=args.hidden_dims)
        self.trgt_mlp = MLPNetwork(obs_dim, num_actions, hidden_dims=args.hidden_dims)

        self.args = args
        if args.use_cuda and th.cuda.is_available():
            self.device = th.device("cuda:0")
            self.eval_mlp.to(self.device)
            self.trgt_mlp.to(self.device)
        else:
            self.device = th.device("cpu")

        self.trgt_mlp.load_state_dict(self.eval_mlp.state_dict())
        self.eval_parameters = list(self.eval_mlp.parameters())
        self.optimizer = th.optim.Adam(self.eval_parameters, lr=args.lr)

        # EMDQN
        self.ec_buffer = []
        for i in range(self.n_actions):
            self.ec_buffer.append(LRUKnn(args.ec_buffer_size, args.ec_latent_dim, env_name=args.env))
        rng = np.random.RandomState(123456)
        self.rp = rng.normal(loc=0, scale=1./np.sqrt(args.ec_latent_dim), size=(args.ec_latent_dim, obs_dim))
        self.qec_watch = []
        self.update_counter = 0
        self.qec_found = 0
        self.sequence = []

        print("Init Algo DQN")

    def select_action(self, obs, epsilon):
        inputs = obs.copy()
        inputs = th.tensor(inputs, dtype=th.float32)
        if self.args.use_cuda and th.cuda.is_available():
            inputs = inputs.to(self.device)

        q_value = self.eval_mlp(inputs)
        if np.random.uniform() < epsilon:
            action = np.random.choice(self.n_actions)
        else:
            action = th.argmax(q_value).item()      # return as INT action index
        return action

    def add_sequence(self, obs, action, reward):
        self.sequence.append([np.array(obs), action, reward])

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
        for u in range(self.n_actions):
            self.ec_buffer[u].update_kdtree()

    def learn(self, batch: dict):
        qec_inputs = self._get_qec_inputs(batch)
        for key in batch.keys():
            if key == 'u':
                batch[key] = th.tensor(batch[key], dtype=th.long)
            else:
                batch[key] = th.tensor(batch[key], dtype=th.float32)
        o, u, o_next = batch['o'], batch['u'], batch['o_next']
        r, terminates = batch['r'], batch['terminates']
        qec_inputs = th.tensor(qec_inputs, dtype=th.float32)
        if self.args.use_cuda and th.cuda.is_available():
            o = o.to(self.device)
            u = u.to(self.device)
            r = r.to(self.device)
            o_next = o_next.to(self.device)
            terminates = terminates.to(self.device)
            qec_inputs = qec_inputs.to(self.device)

        q_values = self.eval_mlp(o)
        q_targets = self.trgt_mlp(o_next)

        q_eval = th.gather(q_values, dim=1, index=u).squeeze(1)
        q_trgt = th.max(q_targets, dim=1)[0]
        target = r + self.args.gamma * (1 - terminates) * q_trgt

        td_error = target.detach() - q_eval
        qec_error = qec_inputs.detach() - q_eval
        loss = td_error.pow(2).mean() + 0.1 * qec_error.pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

    def _get_qec_inputs(self, batch: dict) -> np.ndarray:
        self.update_counter += 1
        o_array, u_array = batch['o'], batch['u']
        o_tensor, u_tensor = th.tensor(o_array, dtype=th.float32), th.tensor(u_array, dtype=th.long)
        if self.args.use_cuda and th.cuda.is_available():
            o_tensor = o_tensor.to(self.device)
            u_tensor = u_tensor.to(self.device)
        qec_values = self.eval_mlp(o_tensor)
        qec_selected = th.gather(qec_values, dim=1, index=u_tensor).squeeze(1)
        qec_selected = np.array(qec_selected.detach()).reshape(self.args.batch_size)  # transfer to array for ec_buffer
        u_array = np.squeeze(u_array, axis=1)   # u_array[i] can then an index
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

        return qec_selected     # shape: [batch_size]

    def update_target(self):
        self.trgt_mlp.load_state_dict(self.eval_mlp.state_dict())
