import pickle

import numpy as np
import torch

from RL.DQN import DQN_agent
from RL.DQNutils import greedy_action
from RL.PPO import PPO_agent
from RL.PPOutils import PPOAgent
from RL.QLearning_multi import QL_agent
from RL.xgb import XGB_agent
from evaluation.calculateInputRange import get_alpha_range


def ql_action():
    alpha_model_type = "DLinear"
    model_V = "1;2;3;4"
    ql_agent = QL_agent(model_type=alpha_model_type, V=model_V)
    f = open("../RL/V1;2;3;4-DLinear.pkl", "rb")
    ql_agent.policy = pickle.load(f)
    f.close()

    counter = 0
    long = []
    short = []
    neutral = []
    for i0 in range(0, ql_agent.num_mid_price_bucket):
        for i1 in range(0, ql_agent.num_mid_price_bucket):
            for i2 in range(0, ql_agent.num_mid_price_bucket):
                for i3 in range(0, ql_agent.num_mid_price_bucket):
                    for i4 in range(0, ql_agent.num_mid_price_bucket):
                        for i5 in range(0, ql_agent.num_mid_price_bucket):
                            for i6 in range(0, ql_agent.num_mid_price_bucket):
                                for i7 in range(0, ql_agent.num_rate_of_return_bucket):
                                    for i8 in range(0, 3):
                                        counter += 1
                                        print(counter, (ql_agent.num_mid_price_bucket ** 7) * ql_agent.num_rate_of_return_bucket * 3)
                                        action = np.argmax(ql_agent.policy[i0][i1][i2][i3][i4][i5][i6][i7][i8])
                                        if action == 0:
                                            # ql_actions.append([i0, i1, i2, i3, i4, i5, i6, i7, 0])
                                            short.append([i0, i1, i2, i3, i4, i5, i6, i7, i8, 0])
                                        elif action == 2:
                                            # ql_actions.append([i0, i1, i2, i3, i4, i5, i6, i7, 2])
                                            long.append([i0, i1, i2, i3, i4, i5, i6, i7, i8, 2])
                                        else:
                                            # ql_actions.append([i0, i1, i2, i3, i4, i5, i6, i7, 1])
                                            neutral.append([i0, i1, i2, i3, i4, i5, i6, i7, i8, 1])

    ql_actions = [long, short, neutral]
    f = open("ql_actions.pkl", "wb")
    pickle.dump(ql_actions, f)
    f.close()


def dqn_action(alpha_bounds, cop_bounds):
    alpha_model_type = "DLinear"
    model_V = "1;2;3;4"
    dqn_agent = DQN_agent(network=[9, 64, 64, 64, 64, 3], model_V=model_V, model_type=alpha_model_type)
    dqn_agent.policy_net = torch.load("../RL/V" + model_V + "-" + alpha_model_type + "-DQN.pt")

    counter = 0
    long = []
    short = []
    neutral = []
    for i0 in range(0, 5):
        for i1 in range(0, 5):
            for i2 in range(0, 5):
                for i3 in range(0, 5):
                    for i4 in range(0, 5):
                        for i5 in range(0, 5):
                            for i6 in range(0, 5):
                                for i7 in range(0, 5):
                                    for i8 in range(0, 3):
                                        counter += 1
                                        print(counter, (5 ** 7) * 5 * 3)
                                        state_0 = alpha_bounds[i0][0]
                                        state_1 = alpha_bounds[i1][1]
                                        state_2 = alpha_bounds[i2][2]
                                        state_3 = alpha_bounds[i3][3]
                                        state_4 = alpha_bounds[i4][4]
                                        state_5 = alpha_bounds[i5][5]
                                        state_6 = alpha_bounds[i6][6]
                                        state_7 = cop_bounds[i7]
                                        state_8 = i8

                                        state = np.array([state_0, state_1, state_2, state_3, state_4, state_5, state_6, state_7, state_8])
                                        state = torch.tensor(state, dtype=torch.float)
                                        action = greedy_action(dqn_agent.policy_net, state)
                                        if action == 0:
                                            # ql_actions.append([i0, i1, i2, i3, i4, i5, i6, i7, 0])
                                            short.append([i0, i1, i2, i3, i4, i5, i6, i7, i8, 0])
                                        elif action == 2:
                                            # ql_actions.append([i0, i1, i2, i3, i4, i5, i6, i7, 2])
                                            long.append([i0, i1, i2, i3, i4, i5, i6, i7, i8, 2])
                                        else:
                                            # ql_actions.append([i0, i1, i2, i3, i4, i5, i6, i7, 1])
                                            neutral.append([i0, i1, i2, i3, i4, i5, i6, i7, i8, 1])

    ql_actions = [long, short, neutral]
    f = open("dqn_actions.pkl", "wb")
    pickle.dump(ql_actions, f)
    f.close()


def ppo_action(alpha_bounds, cop_bounds):
    alpha_model_type = "DLinear"
    model_V = "1;2;3;4"
    ppo_Agent = PPOAgent(network=[9, 64, 64, 64, 3], model_V=model_V, model_type=alpha_model_type)
    ppo_agent = PPO_agent(ppo_Agent)
    ppo_agent.agent.actor = torch.load("../RL/V" + model_V + "-" + alpha_model_type + "-PPO.pt")

    counter = 0
    long = []
    short = []
    neutral = []
    for i0 in range(0, 5):
        for i1 in range(0, 5):
            for i2 in range(0, 5):
                for i3 in range(0, 5):
                    for i4 in range(0, 5):
                        for i5 in range(0, 5):
                            for i6 in range(0, 5):
                                for i7 in range(0, 5):
                                    for i8 in range(0, 3):
                                        counter += 1
                                        print(counter, (5 ** 7) * 5 * 3)
                                        state_0 = alpha_bounds[i0][0]
                                        state_1 = alpha_bounds[i1][1]
                                        state_2 = alpha_bounds[i2][2]
                                        state_3 = alpha_bounds[i3][3]
                                        state_4 = alpha_bounds[i4][4]
                                        state_5 = alpha_bounds[i5][5]
                                        state_6 = alpha_bounds[i6][6]
                                        state_7 = cop_bounds[i7]
                                        state_8 = i8

                                        state = np.array([state_0, state_1, state_2, state_3, state_4, state_5, state_6, state_7, state_8])
                                        state = torch.tensor(state, dtype=torch.float)
                                        action = ppo_agent.agent.predict(state).item()
                                        if action == 0:
                                            # ql_actions.append([i0, i1, i2, i3, i4, i5, i6, i7, 0])
                                            short.append([i0, i1, i2, i3, i4, i5, i6, i7, i8, 0])
                                        elif action == 2:
                                            # ql_actions.append([i0, i1, i2, i3, i4, i5, i6, i7, 2])
                                            long.append([i0, i1, i2, i3, i4, i5, i6, i7, i8, 2])
                                        else:
                                            # ql_actions.append([i0, i1, i2, i3, i4, i5, i6, i7, 1])
                                            neutral.append([i0, i1, i2, i3, i4, i5, i6, i7, i8, 1])

    ql_actions = [long, short, neutral]
    f = open("ppo_actions.pkl", "wb")
    pickle.dump(ql_actions, f)
    f.close()

def xgb_action(alpha_bounds):
    xgb_agent = XGB_agent()
    xgb_agent.bst.load_model("../RL/bst")

    counter = 0
    long = []
    short = []
    neutral = []
    for i0 in range(0, 5):
        for i1 in range(0, 5):
            for i2 in range(0, 5):
                for i3 in range(0, 5):
                    for i4 in range(0, 5):
                        for i5 in range(0, 5):
                            for i6 in range(0, 5):
                                counter += 1
                                print(counter, (5 ** 7))
                                state_0 = alpha_bounds[i0][0]
                                state_1 = alpha_bounds[i1][1]
                                state_2 = alpha_bounds[i2][2]
                                state_3 = alpha_bounds[i3][3]
                                state_4 = alpha_bounds[i4][4]
                                state_5 = alpha_bounds[i5][5]
                                state_6 = alpha_bounds[i6][6]

                                state = np.array([[state_0, state_1, state_2, state_3, state_4, state_5, state_6]])
                                action = xgb_agent.bst.predict(state).item()
                                if action == 0:
                                    # ql_actions.append([i0, i1, i2, i3, i4, i5, i6, i7, 0])
                                    short.append([i0, i1, i2, i3, i4, i5, i6, 0])
                                elif action == 2:
                                    # ql_actions.append([i0, i1, i2, i3, i4, i5, i6, i7, 2])
                                    long.append([i0, i1, i2, i3, i4, i5, i6, 2])
                                else:
                                    # ql_actions.append([i0, i1, i2, i3, i4, i5, i6, i7, 1])
                                    neutral.append([i0, i1, i2, i3, i4, i5, i6, 1])

    ql_actions = [long, short, neutral]
    f = open("xgb_actions.pkl", "wb")
    pickle.dump(ql_actions, f)
    f.close()

if __name__ == "__main__":
    if True:
        # ql_action()

        f = open("alpha_bounds.pkl", "rb")
        alpha_bounds = pickle.load(f)
        f.close()

        f = open("cop_bounds.pkl", "rb")
        cop_bounds = pickle.load(f)
        f.close()

        # dqn_action(alpha_bounds, cop_bounds)
        # ppo_action(alpha_bounds, cop_bounds)
        xgb_action(alpha_bounds)
