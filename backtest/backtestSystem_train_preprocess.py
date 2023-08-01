import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt


class LOBEnv_train_preprocess(object):
    def __init__(self, data_V: str = "1", model_V: str = "1", model_type: str = "LSTM", deep: bool = False) -> None:
        super().__init__()

        self.alphas_list = []
        self.mid_price_list = []

        Vs = data_V.split(";")
        self.num_dataset = len(Vs)

        for V in Vs:
            if deep:
                f = open("../alphas/" + model_type + "-V" + model_V + "-DQN/V" + V + ".pkl", 'rb')
                alphas = pickle.load(f)
                f.close()
            else:
                f = open("../alphas/" + model_type + "-V" + model_V + "/V" + V + ".pkl", 'rb')
                alphas = pickle.load(f)
                f.close()

            f = open("../collectLOB/V" + V + "/mid_prices.pkl", 'rb')
            mid_price = pickle.load(f)
            f.close()

            if deep:
                self.alphas_list.append(alphas)
                # self.alphas_list.append(torch.tensor(alphas, device=torch.device("cuda")))
            else:
                self.alphas_list.append(alphas)

            self.mid_price_list.append(mid_price)

            # mid_price_nparray = np.array(mid_price)
            # plt.ylabel("BTC-TUSD")
            # x = np.arange(1, len(mid_price_nparray) + 1)
            # plt.plot(x, mid_price_nparray)
            # plt.show()

        self.dataset_index = 0
        self.trade_index = 0
        self.pre_mid_price = 0.0
        self.pre_side = 0

        self.account = Account(amount=1.0)

    def reset(self):
        self.dataset_index = 0
        self.trade_index = 0
        self.account = Account(amount=1.0)

        self.pre_mid_price = self.mid_price_list[self.dataset_index][self.trade_index + 100]

        alphas = self.alphas_list[self.dataset_index][self.trade_index]

        return (alphas, 0.0, 0), False

    def step(self, action):
        # the interval of snapshot is 100ms
        # this function returns 100 historical the snapshot

        # consider 100ms delay
        cur_mid_price = self.mid_price_list[self.dataset_index][self.trade_index + 100]
        real_next_mid_price = self.mid_price_list[self.dataset_index][self.trade_index + 110]
        next_mid_price = self.mid_price_list[self.dataset_index][self.trade_index + 109]
        reward = self.account.trade(action - 1, cur_mid_price=cur_mid_price, next_mid_price=real_next_mid_price)

        # next state
        if action != self.pre_side:
            self.pre_mid_price = cur_mid_price
            self.pre_side = action

        rate_of_return = (next_mid_price - self.pre_mid_price) / self.pre_mid_price

        self.trade_index += 10
        alphas = self.alphas_list[self.dataset_index][self.trade_index]

        done = False
        if (self.trade_index + 110 >= len(self.mid_price_list[self.dataset_index])) or (
                self.trade_index + 10 >= len(self.alphas_list[self.dataset_index])):
            if self.dataset_index == self.num_dataset - 1:
                self.dataset_index = 0
                self.trade_index = 0
                self.pre_mid_price = 0.0
                self.pre_side = 0
                done = True
            else:
                self.dataset_index += 1
                self.trade_index = 0
                self.pre_mid_price = self.mid_price_list[self.dataset_index][self.trade_index + 100]
                self.pre_side = 0
        return (alphas, rate_of_return, action), reward, done


class Account(object):
    def __init__(self, amount: float = 1.0) -> None:
        super().__init__()
        self.amount = amount

    def trade(self, side, cur_mid_price, next_mid_price):
        reward = side * (next_mid_price - cur_mid_price) * self.amount
        return reward
