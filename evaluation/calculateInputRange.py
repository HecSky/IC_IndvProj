import pickle
import numpy as np


def get_alpha_range(isDQN: bool = True):
    # data_Vs = ["5"]
    data_Vs = ["5", "6", "7", "8", "9"]
    flag = True
    for data_V in data_Vs:
        if isDQN:
            f = open("../alphas/DLinear-V1;2;3;4-DQN/V" + data_V + ".pkl", 'rb')
        else:
            f = open("../alphas/DLinear-V1;2;3;4/V" + data_V + ".pkl", 'rb')
        alpha = np.array(pickle.load(f))
        if flag:
            alphas = alpha
            flag = False
        else:
            alphas = np.concatenate((alphas, alpha))
        f.close()

    # means = alphas.mean(axis=1)
    # stds = alphas.std(axis=1)
    #
    # bounds = []
    # for i in range(-3, 4):
    #     bound = means + stds * i
    #     bounds.append(bound)

    q = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
    bounds = np.quantile(alphas, q, axis=0)

    f = open("alpha_bounds.pkl", "wb")
    pickle.dump(bounds, f)
    f.close()

    # print(bounds)
    # return bounds

def get_cop_range(isDQN: bool = True):
    f = open("../parameters/rate_of_return_bucket_V1;2;3;4.pkl", "rb")
    cop = pickle.load(f)
    f.close()

    if isDQN:
        f = open('../parameters/normalise_paras_V1;2;3;4_DLinear.pkl', 'rb')
        normalise_paras = pickle.load(f)
        f.close()
        ror_mean = normalise_paras[2]
        ror_std = normalise_paras[3]
        cop = (cop - ror_mean) / ror_std

    f = open("cop_bounds.pkl", "wb")
    pickle.dump(cop, f)
    f.close()

    # return cop


if __name__ == "__main__":
    if True:
        get_alpha_range(True)
        get_cop_range(True)
