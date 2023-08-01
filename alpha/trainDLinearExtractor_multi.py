from datetime import datetime
import os
import sys
sys.path.append(os.path.abspath(".."))
import torch
import torch.nn as nn
from scipy.stats import ks_2samp, anderson_ksamp
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from alpha.LOBDataset_multi import LOBDataset
from alpha.models.DLinear import DLinear


class Configs(object):
    def __init__(self):
        super(Configs, self).__init__()
        self.seq_len = 100
        self.pred_len = 15
        self.individual = False
        self.enc_in = 60
        self.kernel_size = 25


def main(epoch_num):
    log_file = open("../model/DLinear/DLinear_record", mode="a+", encoding="utf-8")
    device = torch.device("cuda")
    model = DLinear(Configs()).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lambda_lr = lambda epoch: max(0.8 ** epoch, 1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    batch_size_train = 512
    dataset_train = LOBDataset(V="1;2;3;4", dataset_type="train", para_name="1;2;3;4")
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size_train, num_workers=8, shuffle=True)
    batch_size_valid = 10240
    dataset_valid = LOBDataset(V="5", dataset_type="valid", para_name="1;2;3;4")
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size_valid, num_workers=8)
    loss_train = []
    loss_valid = []
    sum_ks_stat_list = []
    sum_ad_stat_list = []
    real_mid_price_returns = None
    filled = False
    no_improved = 0
    stop_threshold = 10

    for epoch_ndx in range(1, epoch_num + 1):
        print("{:d}/{:d}".format(epoch_ndx, epoch_num))
        print("{:d}/{:d}".format(epoch_ndx, epoch_num), file=log_file)
        print("lr: " + str(scheduler.get_last_lr()[-1]))
        print("lr: " + str(scheduler.get_last_lr()[-1]), file=log_file)
        total_loss = 0.0
        acc_num = 0
        model.train()
        for batch_ndx, sample in enumerate(dataloader_train):
            LOB = sample[0].to(device)
            mid_price_return = sample[1].to(device)
            output = model(LOB)
            loss = loss_fn(output, mid_price_return)
            # print(loss.item())
            total_loss += abs(loss.item() * len(mid_price_return))
            acc = torch.where((((output > 0) & (mid_price_return > 0)) |
                                ((output < 0) & (mid_price_return < 0)) |
                                (mid_price_return == 0)
                                ), True, False)
            acc_num += torch.sum(acc).item()
            # for name, params in model.named_parameters():
            #     print(params.grad)
            #     break
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            optimizer.step()
        scheduler.step()
        total_loss = total_loss / len(dataset_train)
        loss_train.append(total_loss)
        print("Train-{:d}: {:.16f}".format(epoch_ndx, total_loss))
        print("Train-{:d}: {:.16f}".format(epoch_ndx, total_loss), file=log_file)
        print("Train-{:d}: Acc:{:.16f}".format(epoch_ndx, acc_num/(len(dataset_train)*7)))
        print("Train-{:d}: Acc:{:.16f}".format(epoch_ndx, acc_num/(len(dataset_train)*7)), file=log_file)

        total_loss = 0.0
        acc_num = 0
        outputs = None
        sum_ks_stat = 0.0
        sum_ad_stat = 0.0
        with torch.no_grad():
            model.eval()
            for batch_ndx, sample in enumerate(dataloader_valid):
                LOB = sample[0].to(device)
                mid_price_return = sample[1].to(device)
                output = model(LOB)
                loss = loss_fn(output, mid_price_return)
                total_loss += abs(loss.item() * len(mid_price_return))
                acc = torch.where((((output > 0) & (mid_price_return > 0)) |
                                   ((output < 0) & (mid_price_return < 0)) |
                                   ((torch.abs(output) < 1e-6) & (mid_price_return == 0))
                                   ), 1, 0)
                acc_num += torch.sum(acc).item()

                if type(real_mid_price_returns) == type(None):
                    real_mid_price_returns = mid_price_return
                elif not filled:
                    real_mid_price_returns = torch.cat((real_mid_price_returns, mid_price_return), dim=0)
                    
                if type(outputs) == type(None):
                    outputs = output
                else:
                    outputs = torch.cat((outputs, output), dim=0)
            if not filled:
                real_mid_price_returns = torch.transpose(real_mid_price_returns, dim0=0, dim1=1).to("cpu").numpy()
            filled = True
            predicted_mid_price_returns = torch.transpose(outputs, dim0=0, dim1=1).to("cpu").numpy()
            for i in range(len(predicted_mid_price_returns)):
                ks_stat, ks_p_value = ks_2samp(predicted_mid_price_returns[i], real_mid_price_returns[i])
                print("KS statistic:", ks_stat, ",p-value:", ks_p_value)
                print("KS statistic:", ks_stat, ",p-value:", ks_p_value, file=log_file)
                ad_stat, ad_critical_values, ad_significance_levels = anderson_ksamp([predicted_mid_price_returns[i], real_mid_price_returns[i]])
                print("ad_stat:", ad_stat, ",ad_critical_values:", ad_critical_values, ",ad_significance_levels:", ad_significance_levels)
                print("ad_stat:", ad_stat, ",ad_critical_values:", ad_critical_values, ",ad_significance_levels:", ad_significance_levels, file=log_file)
                sum_ks_stat += ks_stat
                sum_ad_stat += ad_stat
            print("sum_ks_stat:", sum_ks_stat, ",sum_ad_stat:", sum_ad_stat)
            print("sum_ks_stat:", sum_ks_stat, ",sum_ad_stat:", sum_ad_stat, file=log_file)
            sum_ks_stat_list.append(sum_ks_stat)
            sum_ad_stat_list.append(sum_ad_stat)

        total_loss = total_loss / len(dataset_valid)
        loss_valid.append(total_loss)
        print("Valid-{:d}: {:.16f}".format(epoch_ndx, total_loss))
        print("Valid-{:d}: {:.16f}".format(epoch_ndx, total_loss), file=log_file)
        print("Valid-{:d}: Acc:{:.16f}".format(epoch_ndx, acc_num/(len(dataset_valid)*7)))
        print("Valid-{:d}: Acc:{:.16f}".format(epoch_ndx, acc_num/(len(dataset_valid)*7)), file=log_file)

        torch.save(model, "../model/DLinear/DLinear-{:d}.pt".format(epoch_ndx))

        if min(sum_ks_stat_list) < sum_ks_stat and min(sum_ad_stat_list) < sum_ad_stat:
            no_improved += 1
            if no_improved > stop_threshold:
                break
        else:
            no_improved = 0




if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    main(100)
