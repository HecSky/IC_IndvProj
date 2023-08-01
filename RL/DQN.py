import pickle
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from torch import nn

from RL.DQNutils import DQN, ReplayBuffer, epsilon_greedy, loss, update_target, greedy_action
from backtest.backtestSystem_train_preprocess import LOBEnv_train_preprocess
from backtest.backtestSystem_valid_preprocess import LOBEnv_valid_preprocess


class DQN_agent(object):
    def __init__(self, network: list,
                 gamma: float = 0.95,
                 initial_epsilon: float = 1.0,
                 min_epsilon: float = 0.01,
                 epsilon_decay: float = 0.9725,
                 lr_initial: float = 1e-4,
                 lr_decay: float = 0.97,
                 lr_min: int = 5e-1,
                 update_freq: int = 3,
                 batch_size: int = 64,
                 memory: int = 550000,
                 model_V: str = "1;2;3;4",
                 model_type: str = "DLinear"
                 ) -> None:
        super().__init__()
        self.device_name = "cpu"
        self.device = torch.device(self.device_name)
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.lr = lr_initial
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.replayBuffer = ReplayBuffer(memory)
        self.policy_net = DQN(network).to(self.device)
        self.target_net = DQN(network).to(self.device)
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.lr)
        lambda_lr = lambda epoch: max(lr_decay ** epoch, lr_min)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)
        f = open('../parameters/normalise_paras_V' + model_V + "_" + model_type + '.pkl', 'rb')
        normalise_paras = pickle.load(f)
        f.close()
        self.ror_mean = normalise_paras[2].item()
        self.ror_std = normalise_paras[3].item()

    def solve(self, env, episode_num, model_type):
        rewards = []
        epsilon = self.initial_epsilon
        for episode in range(1, episode_num + 1):
            print("{:d}/{:d}".format(episode, episode_num))
            if episode % 100 == 0:
                torch.save(self.policy_net, "V1;2;3;4-" + model_type + "-DQN.pt")

            state, done = env.reset()

            if self.device_name == "cpu":
                state_ = state[0][:]
                state_.append((state[1] - self.ror_mean) / self.ror_std)
                state_.append(state[2])
                state = torch.tensor(state_)
            else:
                state = torch.cat(
                    (state[0], torch.tensor([((state[1] - self.ror_mean) / self.ror_std), state[2]]).to(self.device)),
                    dim=0)

            epsilon = epsilon * self.epsilon_decay
            if epsilon < self.min_epsilon:
                epsilon = self.min_epsilon

            total_reward = 0.0
            positive_reward = 0.0
            abs_total_reward = 0.0
            long_num = 0
            mid_num = 0
            short_num = 0
            counter = 0
            while not done:
                if env.trade_index % 100000 == 0:
                    print("Episode{:d}:{:d}".format(episode, int(env.trade_index)))

                action = epsilon_greedy(epsilon, self.policy_net, state)

                if action == 0:
                    short_num += 1
                elif action == 1:
                    mid_num += 1
                else:
                    long_num += 1

                next_state, reward, done = env.step(action)
                if self.device_name == "cpu":
                    state_ = next_state[0][:]
                    state_.append((next_state[1] - self.ror_mean) / self.ror_std)
                    state_.append(next_state[2])
                    next_state = torch.tensor(state_)
                else:
                    next_state = torch.cat((next_state[0], torch.tensor(
                        [((next_state[1] - self.ror_mean) / self.ror_std), next_state[2]]).to(self.device)), dim=0)

                total_reward += reward
                abs_total_reward += abs(reward)
                if reward > 0:
                    positive_reward += reward

                reward = torch.tensor([reward]).to(self.device)
                action = torch.tensor([action]).to(self.device)
                self.replayBuffer.push([state, action, next_state, reward, torch.tensor([done]).to(self.device)])

                state = next_state
                counter += 1
                if len(self.replayBuffer.buffer) >= self.batch_size and counter == self.batch_size:
                    counter = 0
                    transitions = self.replayBuffer.sample(self.batch_size)
                    state_batch, action_batch, next_state_batch, reward_batch, dones = (torch.stack(x) for x in
                                                                                        zip(*transitions))

                    mse_loss = loss(self.policy_net, self.target_net, state_batch, action_batch, reward_batch,
                                    next_state_batch, dones, self.gamma, DDQN=True)

                    self.optimizer.zero_grad()
                    mse_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1, norm_type=2)
                    self.optimizer.step()

            self.scheduler.step()
            print("lr: " + str(self.scheduler.get_last_lr()[-1]))
            print("epsilon: ", str(epsilon))
            rewards.append(total_reward)

            if episode % self.update_freq == 0:
                update_target(self.target_net, self.policy_net)

            print("Train{:d}-Reward:{:.8f}-Accuracy:{:.8f}".format(episode, total_reward,
                                                                   positive_reward / abs_total_reward))
            print("Long:{:d}, Mid:{:d}, Short:{:d}".format(long_num, mid_num, short_num))
            print("Average_reward:{:.8f}".format(sum(rewards) / len(rewards)))

    def test(self, env, model_V: str = "", data_V: str = "", model_type: str = "", return_res: bool = False):
        with torch.no_grad():
            self.policy_net.eval()
            state, done = env.reset()

            if self.device_name == "cpu":
                state_ = state[0][:]
                state_.append((state[1] - self.ror_mean) / self.ror_std)
                state_.append(state[2])
                state = torch.tensor(state_)
            else:
                state = torch.cat(
                    (state[0], torch.tensor([((state[1] - self.ror_mean) / self.ror_std), state[2]]).to(self.device)),
                    dim=0)

            total_reward = 0.0
            abs_total_reward = 0.0
            positive_reward = 0.0
            last_action = 1
            action_duration = 1
            action_durations = []
            action_rewards = []
            action_reward = 0
            rewards = [0]
            actions = []
            trade_num = 0

            long_num = 0
            mid_num = 0
            short_num = 0
            while not done:
                if env.index % 100000 == 0:
                    print("Index:{:d}".format(int(env.index)))

                action = greedy_action(self.policy_net, state)

                # if action == 0:
                #     action = 1

                actions.append(action)

                if action == 0:
                    short_num += 1
                elif action == 1:
                    mid_num += 1
                else:
                    long_num += 1

                next_state, reward, done = env.step(action)

                if self.device_name == "cpu":
                    state_ = next_state[0][:]
                    state_.append((next_state[1] - self.ror_mean) / self.ror_std)
                    state_.append(next_state[2])
                    state = torch.tensor(state_)
                else:
                    state = torch.cat((next_state[0], torch.tensor(
                        [((next_state[1] - self.ror_mean) / self.ror_std), next_state[2]]).to(self.device)), dim=0)

                total_reward += reward
                abs_total_reward += abs(reward)
                rewards.append(total_reward)

                if last_action == action:
                    action_duration += 1
                    action_reward += reward
                else:
                    action_durations.append(action_duration)
                    action_duration = 1
                    last_action = action
                    trade_num += 1
                    action_rewards.append(action_reward)
                    action_reward = reward

                if reward > 0:
                    positive_reward += reward

            if return_res:
                return rewards, trade_num, np.array(action_durations).mean().item(), action_rewards
            else:
                rewards = np.array(rewards)
                xs = np.arange(0, len(rewards))
                color = []
                for i in range(0, len(rewards) - 1):
                    if actions[i] == 0:
                        color.append("#008000")
                    elif actions[i] == 1:
                        color.append("#0000FF")
                    elif actions[i] == 2:
                        color.append("#FF0000")
                    else:
                        color.append("#FFFFFF")

                points = np.array([xs, rewards]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                line_collection = LineCollection(segments, color=color)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.set_xlim(min(xs) - (max(xs) - min(xs)) / 20, max(xs) + (max(xs) - min(xs)) / 20)
                ax.set_ylim(min(rewards) - (max(rewards) - min(rewards)) / 20,
                            max(rewards) + (max(rewards) - min(rewards)) / 20)
                ax.add_collection(line_collection)
                plt.ylabel("PnL")
                plt.xlabel("Time")
                plt.savefig("../figure/" + model_type + "-V" + model_V + "_" + data_V + "_DQN.png")
                plt.show()

                print("Valid-Reward:{:.8f}-Accuracy:{:.8f}".format(total_reward, positive_reward / abs_total_reward))
                print("Long:{:d}, Mid:{:d}, Short:{:d}".format(long_num, mid_num, short_num))
                action_durations = np.array(action_durations)
                print("Average side duration:{:8f}".format(action_durations.mean()))


if __name__ == "__main__":
    torch.set_printoptions(precision=8)

    alpha_model_type = "DLinear"
    model_V = "1;2;3;4"

    dqn_agent = DQN_agent(network=[9, 64, 64, 64, 64, 3], model_V=model_V, model_type=alpha_model_type)

    env_train = LOBEnv_train_preprocess(data_V="1;2;3;4", model_V=model_V, model_type=alpha_model_type, deep=True)
    dqn_agent.solve(env_train, 200, alpha_model_type)

    dqn_agent.policy_net = torch.load("V" + model_V + "-" + alpha_model_type + "-DQN.pt")

    data_Vs = ["5", "6", "7", "8", "9"]
    for d_V in data_Vs:
        env_valid = LOBEnv_valid_preprocess(data_V=d_V, model_V=model_V, model_type=alpha_model_type, deep=True)
        dqn_agent.test(env_valid, data_V=d_V, model_V=model_V, model_type=alpha_model_type)
