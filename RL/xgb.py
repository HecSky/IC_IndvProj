import pickle
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from torch import nn
from xgboost import plot_tree, to_graphviz
from xgboost import XGBClassifier
from backtest.backtestSystem_train_preprocess import LOBEnv_train_preprocess
from backtest.backtestSystem_valid_preprocess import LOBEnv_valid_preprocess


class XGB_agent(object):
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1) -> None:
        super().__init__()
        self.bst = XGBClassifier(n_estimators=n_estimators,
                                 max_depth=6,
                                 learning_rate=learning_rate,
                                 num_class=3,
                                 objective='multi:softmax',
                                 eval_metric="mlogloss")

    def train(self, model_type: str, model_V: str, data_V: str):
        alphas_list = []
        label_list = []

        Vs = data_V.split(";")

        trading_interval = 10

        for V in Vs:
            f = open("../alphas/" + model_type + "-V" + model_V + "-DQN/V" + V + ".pkl", 'rb')
            alphas = pickle.load(f)
            alphas = np.array(alphas, dtype=np.float64)
            f.close()

            f = open("../collectLOB/V" + V + "/mid_prices.pkl", 'rb')
            mid_price = pickle.load(f)[100:][:len(alphas)]
            mid_price = np.array(mid_price, dtype=np.float64)
            f.close()

            rs1 = mid_price[trading_interval:] - mid_price[:-trading_interval]
            rs1 = rs1[1:]
            rs2 = mid_price[int(trading_interval*1.3):] - mid_price[:-int(trading_interval*1.3)]
            rs2 = rs2[1:]
            alphas = alphas[:len(rs2)]
            label = np.zeros_like(rs2, dtype=np.int32)
            for i, alpha in enumerate(alphas):
                if rs1[i] > 0 > rs2[i] or rs2[i] > 0 > rs1[i]:
                    label[i] = 1
                elif rs1[i] > 0 or rs2[i] > 0:
                    label[i] = 2

            alphas_list.append(alphas)
            label_list.append(label)

        alphas_array = np.concatenate(alphas_list, axis=0)
        label_array = np.concatenate(label_list, axis=0, dtype=np.int32)

        self.bst.fit(alphas_array, label_array)

        fig, ax = plt.subplots(figsize=(20, 35))
        plt.subplots_adjust(top=0.99, bottom=0.01, right=0.99, left=0.01)
        plot_tree(self.bst, rankdir='LR', ax=ax)
        plt.savefig("../figure/xgboost_tree.png", bbox_inches="tight")
        plt.show()

        self.bst.save_model("bst")

    def test(self, env, model_V: str = "", data_V: str = "", model_type: str = "", return_res: bool = False):
        state, done = env.reset()

        state = np.array([state[0]], dtype=np.float64)

        total_reward = 0.0
        abs_total_reward = 0.0
        positive_reward = 0.0
        last_action = 1
        action_duration = 1
        action_durations = []
        action_rewards = []
        action_reward = 0
        rewards = [0]
        rewards_list = [0.0]
        actions = []
        trade_num = 0
        # test
        long_num = 0
        mid_num = 0
        short_num = 0
        while not done:
            if env.index % 100000 == 0:
                print("Index:{:d}".format(int(env.index)))

            action = self.bst.predict(state).item()

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

            state = np.array([next_state[0]], dtype=np.float64)

            total_reward += reward
            abs_total_reward += abs(reward)
            rewards.append(total_reward)
            rewards_list.append(reward)

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
            print("Valid-Reward:{:.8f}-Accuracy:{:.8f}".format(total_reward, positive_reward / abs_total_reward))
            print("Long:{:d}, Mid:{:d}, Short:{:d}".format(long_num, mid_num, short_num))
            action_durations = np.array(action_durations)
            print("Average side duration:{:8f}".format(action_durations.mean()))
            return rewards, trade_num, np.array(action_durations), action_rewards, rewards_list
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
            plt.savefig("../figure/" + model_type + "-V" + model_V + "_" + data_V + "_XGBOOST.png")
            plt.show()

            print("Valid-Reward:{:.8f}-Accuracy:{:.8f}".format(total_reward, positive_reward / abs_total_reward))
            print("Long:{:d}, Mid:{:d}, Short:{:d}".format(long_num, mid_num, short_num))
            action_durations = np.array(action_durations)
            print("Average side duration:{:8f}".format(action_durations.mean()))


if __name__ == "__main__":
    torch.set_printoptions(precision=8)

    alpha_model_type = "DLinear"
    model_V = "1;2;3;4"

    xgb_agent = XGB_agent()

    xgb_agent.train(alpha_model_type, model_V, model_V)

    xgb_agent.bst.load_model("bst")

    # fig, ax = plt.subplots(figsize=(20, 35))
    # plt.subplots_adjust(top=0.99, bottom=0.01, right=0.99, left=0.01)
    # plot_tree(xgb_agent.bst, num_trees=297, rankdir='LR', ax=ax)
    # # plt.savefig("../figure/xgboost_tree.png", bbox_inches="tight")
    # plt.show()

    # fi = xgb_agent.bst.feature_importances_
    # fig, ax = plt.subplots()
    # plt.title("Feature importances")
    # plt.bar(range(len(fi)), fi)
    # plt.savefig("../figure/xgboost_tree_fi.png")
    # plt.show()

    data_Vs = ["5", "6", "7", "8", "9"]
    # data_Vs = ["5"]
    for d_V in data_Vs:
        env_valid = LOBEnv_valid_preprocess(data_V=d_V, model_V=model_V, model_type=alpha_model_type, deep=True)
        xgb_agent.test(env_valid, data_V=d_V, model_V=model_V, model_type=alpha_model_type)

    # lrs = [0.01, 0.05]
    # for lr in lrs:
    #     print("lr:", lr)
    #     R = []
    #     for i in range(200, 0, -4):
    #         print("n_estimators:", i)
    #         xgb_agent = XGB_agent(n_estimators=i, learning_rate=lr)
    #
    #         xgb_agent.train(alpha_model_type, model_V, model_V)
    #
    #         xgb_agent.bst.load_model("bst")
    #
    #         # data_Vs = ["5", "6", "7", "8", "9"]
    #         data_Vs = ["5"]
    #         for d_V in data_Vs:
    #             env_valid = LOBEnv_valid_preprocess(data_V=d_V, model_V=model_V, model_type=alpha_model_type, deep=True)
    #             rewards_, A, B, C = xgb_agent.test(env_valid, data_V=d_V, model_V=model_V, model_type=alpha_model_type, return_res=True)
    #             R.append(rewards_[-1])
    #
    #     R = np.array(R)
    #     plt.title(str(lr))
    #     x = np.arange(1, len(R) + 1)
    #     plt.plot(x, R)
    #     plt.show()
