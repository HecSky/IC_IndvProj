import pickle

import numpy as np
import torch

from matplotlib import pyplot as plt


class LOBEnv_valid_preprocess(object):
    def __init__(self, data_V: str = "1", model_V: str = "1", model_type: str = "LSTM", deep: bool = False) -> None:
        super().__init__()

        if deep:
            f = open("../alphas/" + model_type + "-V" + model_V + "-DQN/V" + data_V + ".pkl", 'rb')
            self.alphas = pickle.load(f)
            # self.alphas = torch.tensor(self.alphas, device=torch.device("cuda"))
            f.close()
        else:
            f = open("../alphas/" + model_type + "-V" + model_V + "/V" + data_V + ".pkl", 'rb')
            self.alphas = pickle.load(f)
            f.close()



        f = open("../collectLOB/V" + str(data_V) + "/mid_prices.pkl", 'rb')
        self.mid_price = pickle.load(f)
        f.close()

        # mid_price_nparray = np.array(self.mid_price)
        # plt.ylabel("BTC-TUSD")
        # x = np.arange(1, len(mid_price_nparray) + 1)
        # plt.plot(x, mid_price_nparray)
        # plt.show()

        self.index = 0
        self.pre_mid_price = 0.0
        self.pre_side = 1

        self.account = Account(amount=0.1)

    def reset(self):
        self.index = 0
        self.account = Account(amount=0.1)

        self.pre_mid_price = self.mid_price[self.index + 100]

        done = False
        alphas = self.alphas[self.index]
        return (alphas, 0.0, 1), done

    def step(self, action):
        # the interval of snapshot is 100ms
        # this function returns 100 historical the snapshot

        # consider 100ms delay
        cur_mid_price = self.mid_price[self.index + 100]
        real_next_mid_price = self.mid_price[self.index + 110]
        next_mid_price = self.mid_price[self.index + 109]
        reward = self.account.trade(action - 1, cur_mid_price=cur_mid_price, next_mid_price=real_next_mid_price)

        # next state
        if action != self.pre_side:
            self.pre_mid_price = cur_mid_price
            self.pre_side = action

        rate_of_return = (next_mid_price - self.pre_mid_price) / self.pre_mid_price

        self.index += 10
        alphas = self.alphas[self.index]

        done = False
        if self.index + 110 >= len(self.mid_price) or self.index + 10 >= len(self.alphas):
            done = True
            self.index = 0
            self.pre_mid_price = 0.0
            self.pre_side = 1
        return (alphas, rate_of_return, action), reward, done


class Account(object):
    def __init__(self, amount: float = 1.0) -> None:
        super().__init__()
        self.amount = amount

    def trade(self, side, cur_mid_price, next_mid_price):
        cur_mid_price = round(cur_mid_price, 3)
        next_mid_price = round(next_mid_price, 3)
        reward = side * (next_mid_price - cur_mid_price) * self.amount
        reward = round(reward, 3)
        return reward
