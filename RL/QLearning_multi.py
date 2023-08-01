import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from backtest.backtestSystem_train_preprocess import LOBEnv_train_preprocess

from backtest.backtestSystem_valid_preprocess import LOBEnv_valid_preprocess


class QL_agent(object):
    def __init__(self, V: str,
                 model_type: str,
                 gamma: float = 0.95,
                 initial_epsilon: float = 1.0,
                 min_epsilon: float = 0.005,
                 epsilon_decay: float = 0.9725,
                 initial_lr: float = 0.1,
                 min_lr: float = 0.0005,
                 lr_decay: float = 0.955) -> None:
        super().__init__()
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.initial_lr = initial_lr
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.policy = None
        self.mid_price_bucket = None
        self.rate_of_return_bucket = None
        f = open("../parameters/mid_price_bucket_V" + V + "_" + model_type + ".pkl", "rb")
        self.mid_price_bucket = pickle.load(f)
        f.close()
        f = open("../parameters/rate_of_return_bucket_V" + V + ".pkl", "rb")
        self.rate_of_return_bucket = pickle.load(f)
        f.close()
        self.num_mid_price_bucket = len(self.mid_price_bucket[0]) + 1
        self.num_rate_of_return_bucket = len(self.rate_of_return_bucket) + 1

    def value_to_bucket(self, value, index):
        for i in range(self.num_mid_price_bucket - 1):
            if value < self.mid_price_bucket[index][i]:
                return i
        return self.num_mid_price_bucket - 1

    def mid_price_to_bucket(self, mid_price):
        i0 = self.value_to_bucket(mid_price[0], 0)
        i1 = self.value_to_bucket(mid_price[1], 1)
        i2 = self.value_to_bucket(mid_price[2], 2)
        i3 = self.value_to_bucket(mid_price[3], 3)
        i4 = self.value_to_bucket(mid_price[4], 4)
        i5 = self.value_to_bucket(mid_price[5], 5)
        i6 = self.value_to_bucket(mid_price[6], 6)
        return i0, i1, i2, i3, i4, i5, i6

    def rate_of_return_to_bucket(self, rate_of_return):
        for i in range(self.num_rate_of_return_bucket - 1):
            if rate_of_return < self.rate_of_return_bucket[i]:
                return i
        return self.num_rate_of_return_bucket - 1

    def epsilon_greedy(self, epsilon, Q, i0, i1, i2, i3, i4, i5, i6, i7, i8):
        policy = np.zeros(3)
        policy += epsilon / 3
        optimal_action = np.argmax(Q[i0][i1][i2][i3][i4][i5][i6][i7][i8])
        policy[optimal_action] = 1.0 - epsilon + epsilon / 3
        return policy

    def solve(self, env, episode_num):
        Q = np.random.rand(self.num_mid_price_bucket, self.num_mid_price_bucket, self.num_mid_price_bucket,
                           self.num_mid_price_bucket, self.num_mid_price_bucket, self.num_mid_price_bucket,
                           self.num_mid_price_bucket, self.num_rate_of_return_bucket, 3, 3)
        epsilon = self.initial_epsilon
        lr = self.initial_lr
        rewards = []
        # policy = [1/3, 1/3, 1/3]
        for episode in range(1, episode_num + 1):
            print("{:d}/{:d}".format(episode, episode_num))
            epsilon = epsilon * self.epsilon_decay
            lr = lr * self.lr_decay
            if epsilon < self.min_epsilon:
                epsilon = self.min_epsilon
            if lr < self.min_lr:
                lr = self.min_lr
            state, done = env.reset()
            i0, i1, i2, i3, i4, i5, i6 = self.mid_price_to_bucket(state[0])
            i7 = self.rate_of_return_to_bucket(state[1])
            i8 = state[2]
            total_reward = 0.0
            positive_reward = 0.0
            abs_total_reward = 0.0
            last_action = 1
            action_duration = 1
            action_durations = []
            long_num = 0
            mid_num = 0
            short_num = 0
            while not done:
                if env.trade_index % 100000 == 0:
                    print("Episode{:d}:{:d}".format(episode, int(env.trade_index)))
                policy = self.epsilon_greedy(epsilon, Q, i0, i1, i2, i3, i4, i5, i6, i7, i8)
                action = np.random.choice(3, 1, p=policy)[0]
                if action == 0:
                    short_num += 1
                elif action == 1:
                    mid_num += 1
                else:
                    long_num += 1

                if last_action == action:
                    action_duration += 1
                else:
                    action_durations.append(action_duration)
                    action_duration = 1
                    last_action = action

                next_state, reward, done = env.step(action)

                ni0, ni1, ni2, ni3, ni4, ni5, ni6 = self.mid_price_to_bucket(next_state[0])
                ni7 = self.rate_of_return_to_bucket(next_state[1])
                ni8 = next_state[2]
                policy = self.epsilon_greedy(epsilon, Q, ni0, ni1, ni2, ni3, ni4, ni5, ni6, ni7, ni8)
                next_action = np.argmax(policy)
                Q[i0][i1][i2][i3][i4][i5][i6][i7][i8][action] = Q[i0][i1][i2][i3][i4][i5][i6][i7][i8][
                                                                    action] + lr * (reward + self.gamma *
                                                                                    Q[ni0][ni1][ni2][ni3][ni4][ni5][
                                                                                        ni6][ni7][ni8][
                                                                                        next_action] -
                                                                                    Q[i0][i1][i2][i3][i4][i5][i6][i7][
                                                                                        i8][action])
                i0, i1, i2, i3, i4, i5, i6, i7, i8 = ni0, ni1, ni2, ni3, ni4, ni5, ni6, ni7, ni8
                total_reward += reward
                abs_total_reward += abs(reward)
                if reward > 0:
                    positive_reward += reward
            rewards.append(total_reward)
            print("Train{:d}-Reward:{:.8f}-Accuracy:{:.8f}".format(episode, total_reward,
                                                                   positive_reward / abs_total_reward))
            print("Long:{:d}, Mid:{:d}, Short:{:d}".format(long_num, mid_num, short_num))
            print("Average_reward:{:.8f}".format(sum(rewards) / len(rewards)))
            action_durations = np.array(action_durations)
            print("Average side duration:{:8f}".format(action_durations.mean()))
            # if episode != 0 and episode % 50 == 0:
            #     self.create_policy(Q=Q)
            #     f = open('qlearning_policy.pkl', 'wb')
            #     pickle.dump(self.policy, f)
            #     f.close()

        self.create_policy(Q=Q)

    def create_policy(self, Q):
        # optimal policy
        self.policy = np.zeros((self.num_mid_price_bucket, self.num_mid_price_bucket, self.num_mid_price_bucket,
                                self.num_mid_price_bucket, self.num_mid_price_bucket, self.num_mid_price_bucket,
                                self.num_mid_price_bucket, self.num_rate_of_return_bucket, 3, 3))
        for index_0 in range(0, self.num_mid_price_bucket):
            for index_1 in range(0, self.num_mid_price_bucket):
                for index_2 in range(0, self.num_mid_price_bucket):
                    for index_3 in range(0, self.num_mid_price_bucket):
                        for index_4 in range(0, self.num_mid_price_bucket):
                            for index_5 in range(0, self.num_mid_price_bucket):
                                for index_6 in range(0, self.num_mid_price_bucket):
                                    for index_7 in range(0, self.num_rate_of_return_bucket):
                                        for index_8 in range(0, 3):
                                            self.policy[index_0][index_1][index_2][index_3][index_4][index_5][index_6][
                                                index_7][index_8][
                                                np.argmax(
                                                    Q[index_0][index_1][index_2][index_3][index_4][index_5][index_6][
                                                        index_7][index_8])] = 1.0

    def test(self, env, model_V: str = "", data_V: str = "", model_type: str = "", return_res: bool = False):
        state, done = env.reset()
        i0, i1, i2, i3, i4, i5, i6 = self.mid_price_to_bucket(state[0])
        i7 = self.rate_of_return_to_bucket(state[1])
        i8 = state[2]
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
            action = np.argmax(self.policy[i0][i1][i2][i3][i4][i5][i6][i7][i8])

            if action == 0:
                action = 1

            actions.append(action)

            if action == 0:
                short_num += 1
            elif action == 1:
                mid_num += 1
            else:
                long_num += 1

            next_state, reward, done = env.step(action)
            i0, i1, i2, i3, i4, i5, i6 = self.mid_price_to_bucket(next_state[0])
            i7 = self.rate_of_return_to_bucket(next_state[1])
            i8 = next_state[2]
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
            plt.savefig("../figure/" + model_type + "-V" + model_V + "_" + data_V + ".png")
            plt.show()

            print("Valid-Reward:{:.8f}-Accuracy:{:.8f}".format(total_reward, positive_reward / abs_total_reward))
            print("Long:{:d}, Mid:{:d}, Short:{:d}".format(long_num, mid_num, short_num))
            action_durations = np.array(action_durations)
            print("Average side duration:{:8f}".format(action_durations.mean()))


if __name__ == "__main__":
    if True:
        torch.set_printoptions(precision=8)
        with torch.no_grad():
            alpha_model_type = "DLinear"
            model_V = "1;2;3;4"
            ql_agent = QL_agent(model_type=alpha_model_type, V=model_V)

            env_train = LOBEnv_train_preprocess(data_V="1;2;3;4", model_V=model_V, model_type=alpha_model_type)
            ql_agent.solve(env_train, 190)

            f = open("V1;2;3;4-" + alpha_model_type + ".pkl", "wb")
            pickle.dump(ql_agent.policy, f)
            f.close()

            f = open("V1;2;3;4-" + alpha_model_type + ".pkl", "rb")
            ql_agent.policy = pickle.load(f)
            f.close()

            data_Vs = ["5", "6", "7", "8", "9"]

            for d_V in data_Vs:
                env_valid = LOBEnv_valid_preprocess(data_V=d_V, model_V=model_V, model_type=alpha_model_type)
                ql_agent.test(env_valid, data_V=d_V, model_V=model_V, model_type=alpha_model_type)
