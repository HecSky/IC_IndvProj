import pickle

import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from alpha.LOBDataset_multi import LOBDataset


def main():
    device = torch.device("cuda:0")
    V = "1;2;3;4"

    Vs = V.split(";")
    rate_of_return = torch.tensor([]).to(device)
    for v in Vs:
        f = open("../collectLOB/V" + str(v) + "/mid_prices.pkl", 'rb')
        mid_price = pickle.load(f)
        f.close()
        mid_price = torch.tensor(mid_price).to(device)
        intervals = [10]
        for interval in intervals:
            interval_rate_of_return = (mid_price[interval:] - mid_price[:-interval]) / mid_price[:-interval]
            rate_of_return = torch.cat([rate_of_return, interval_rate_of_return], dim=0)
    rate_of_return = rate_of_return.to("cpu").numpy()
    plt.hist(rate_of_return, bins=500, density=True)
    plt.ylabel("Sample frequency")
    plt.title("Distribution of rate of return")

    plt.subplots_adjust(wspace=0.3)
    plt.show()


if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    main()
