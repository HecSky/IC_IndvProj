import pickle

import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from scipy.stats import ks_2samp, anderson_ksamp
from alpha.LOBDataset_multi import LOBDataset


def main():
    device = torch.device("cuda:0")
    V = "1;2;3;4"
    model_type = "DLinear"
    model = torch.load("../model/" + model_type + "-V" + V + ".pt").to(device)
    # model = torch.load("../model/" + model_type + "-14.pt").to(device)

    timestamp = False
    dataset_train = LOBDataset(V="1;2;3;4", dataset_type="valid", para_name=V, ts=timestamp)
    dataloader_train = DataLoader(dataset_train, batch_size=10240, num_workers=8, shuffle=True)

    outputs = None
    real_mid_price_returns = None

    with torch.no_grad():
        model.eval()
        for batch_ndx, sample in enumerate(dataloader_train):
            if timestamp:
                LOB = sample[0][0].to(device, non_blocking=True)
                mark = sample[0][1].to(device, non_blocking=True)
                mid_price_return = sample[1].to(device, non_blocking=True)
                output = model(x_enc=LOB, x_mark_enc=mark[:, :100], x_dec=LOB, x_mark_dec=mark[:, -100:])
            else:
                LOB = sample[0].to(device)
                mid_price_return = sample[1].to(device)
                output = model(LOB)
            if outputs == None:
                outputs = output
                real_mid_price_returns = mid_price_return
            else:
                outputs = torch.cat((outputs, output), dim=0)
                real_mid_price_returns = torch.cat((real_mid_price_returns, mid_price_return), dim=0)

    # q = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]).to(device)
    q = torch.tensor([0.05, 0.4, 0.6, 0.95]).to(device)

    print("TMP LOB outputs")
    quantile = torch.quantile(outputs,
                              torch.tensor([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]).to(device), dim=0)
    # print(quantile)
    tmp_mid_price_quantiles = torch.transpose(quantile, dim0=0, dim1=1)
    print(tmp_mid_price_quantiles)

    print("Real label")
    quantile = torch.quantile(real_mid_price_returns, q, dim=0)
    # print(quantile)
    print(torch.transpose(quantile, dim0=0, dim1=1))

    print("LOB outputs")
    quantile = torch.quantile(outputs, q, dim=0)
    # print(quantile)
    mid_price_quantiles = torch.transpose(quantile, dim0=0, dim1=1)
    print(mid_price_quantiles)

    predicted_mid_price_returns = torch.transpose(outputs, dim0=0, dim1=1).to("cpu").numpy()
    # real_mid_price_returns = torch.transpose(real_mid_price_returns, dim0=0, dim1=1).to("cpu").numpy()
    # sum_ks_stat = 0.0
    # sum_ad_stat = 0.0
    # for i in range(len(predicted_mid_price_returns)):
    #     ks_stat, ks_p_value = ks_2samp(predicted_mid_price_returns[i], real_mid_price_returns[i])
    #     print("KS statistic:", ks_stat, ",p-value:", ks_p_value)
    #     ad_stat, ad_critical_values, ad_significance_levels = anderson_ksamp([predicted_mid_price_returns[i], real_mid_price_returns[i]])
    #     print("ad_stat:", ad_stat, ",ad_critical_values:", ad_critical_values, ",ad_significance_levels:", ad_significance_levels)
    #     sum_ks_stat += ks_stat
    #     sum_ad_stat += ad_stat
    # print("sum_ks_stat:", sum_ks_stat, ",sum_ad_stat:", sum_ad_stat)

    # f = open("../parameters/mid_price_bucket_V" + V + "_" + model_type + ".pkl", "wb")
    # pickle.dump(mid_price_quantiles.tolist(), f)
    # f.close()

    # medians = np.median(outputs.to("cpu").numpy(), axis=0).tolist()
    # print(medians)
    # f = open("../parameters/median_V" + V + "_" + model_type + ".pkl", "wb")
    # pickle.dump(medians, f)
    # f.close()

    Vs = V.split(";")
    rate_of_return = torch.tensor([]).to(device)
    for v in Vs:
        f = open("../collectLOB/V" + str(v) + "/mid_prices.pkl", "rb")
        mid_price = pickle.load(f)
        f.close()
        mid_price = torch.tensor(mid_price).to(device)
        intervals = [10, 20, 30, 40]
        for interval in intervals:
            interval_rate_of_return = (mid_price[interval:] - mid_price[:-interval]) / mid_price[:-interval]
            rate_of_return = torch.cat([rate_of_return, interval_rate_of_return], dim=0)
    # q = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]).to(device)
    q = torch.tensor([0.05, 0.4, 0.5, 0.6, 0.95]).to(device)
    # q = torch.tensor([0.05, 0.4, 0.6, 0.95]).to(device)
    rate_of_return = rate_of_return.to("cpu").numpy()
    # q = q.to("cpu").numpy()
    # # rate_of_return_quantile = torch.quantile(rate_of_return, q, dim=0)
    # rate_of_return_quantile = np.quantile(rate_of_return, q, axis=0)
    # print(rate_of_return_quantile)

    # f = open("../parameters/rate_of_return_bucket_V" + V + ".pkl", "wb")
    # pickle.dump(rate_of_return_quantile.tolist(), f)
    # f.close()

    normalise_paras = []
    normalise_paras.append(predicted_mid_price_returns.mean(axis=1))
    normalise_paras.append(predicted_mid_price_returns.std(axis=1))
    normalise_paras.append(rate_of_return.mean())
    normalise_paras.append(rate_of_return.std())
    f = open("../parameters/normalise_paras_V" + V + "_" + model_type + ".pkl", "wb")
    pickle.dump(normalise_paras, f)
    f.close()

if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    main()
