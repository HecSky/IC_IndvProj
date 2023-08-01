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

from alpha.models.FEDformer import FEDformer
from alpha.LOBDataset_multi import LOBDataset



class Configs(object):
    modes = 64
    mode_select = 'random'
    version = 'Fourier'
    # version = 'Wavelets'
    moving_avg = [25]
    L = 3
    base = 'legendre'
    cross_activation = 'tanh'
    seq_len = 100
    label_len = 99
    pred_len = 1
    # label_len = 85
    # pred_len = 15
    output_attention = False
    enc_in = 60
    dec_in = 60
    d_model = 60
    embed_type = 2
    embed = 'timeF'
    freq = 'ms'
    dropout = 0.1
    factor = 1
    n_heads = 8
    d_ff = 256
    e_layers = 1
    d_layers = 1
    c_out = 60
    num_alphas = 7
    activation = 'prelu'
    # wavelet = 0


def main(epoch_num):
    log_file = open("../model/FEDformer/FEDformer_record", mode="a+", encoding="utf-8")
    device = torch.device("cuda")
    model = FEDformer(Configs()).to(device)
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    lambda_lr = lambda epoch: max(0.99 ** epoch, 2e-1)
    last_epoch = 0

    # checkpoint = torch.load("../model/backup/Checkpoint_FEDformer.pth")
    # last_epoch = checkpoint["epoch_ndx"]
    # model.load_state_dict(checkpoint["model"])
    # optimizer.load_state_dict(checkpoint["optimizer"])

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr, last_epoch=last_epoch - 1)

    batch_size_train = 512
    dataset_train = LOBDataset(V="1;2;3;4", dataset_type="train", para_name="1;2;3;4", ts=True)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size_train, num_workers=8, shuffle=True)
    batch_size_valid = 4096
    dataset_valid = LOBDataset(V="5", dataset_type="valid", para_name="1;2;3;4", ts=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size_valid, num_workers=8)
    loss_train = []
    loss_valid = []
    sum_ks_stat_list = []
    sum_ad_stat_list = []
    real_mid_price_returns = None
    filled = False
    no_improved = 0
    stop_threshold = 10

    for epoch_ndx in range(last_epoch + 1, epoch_num + 1):
        print("{:d}/{:d}".format(epoch_ndx, epoch_num))
        print("{:d}/{:d}".format(epoch_ndx, epoch_num), file=log_file)
        print("lr: " + str(scheduler.get_last_lr()[-1]))
        print("lr: " + str(scheduler.get_last_lr()[-1]), file=log_file)
        total_loss = 0.0
        acc_num = 0
        model.train()
        for batch_ndx, sample in enumerate(dataloader_train):
            # print(batch_ndx)
            LOB = sample[0][0].to(device, non_blocking=True)
            mark = sample[0][1].to(device, non_blocking=True)
            mid_price_return = sample[1].to(device, non_blocking=True)
            output = model(x_enc=LOB, x_mark_enc=mark[:, :100], x_dec=LOB, x_mark_dec=mark[:, -100:])
            # output = model(x_enc=LOB, x_mark_enc=mark[:, :100], x_dec=LOB[:, -85:], x_mark_dec=mark[:, -100:])
            loss = loss_fn(output, mid_price_return)
            total_loss += loss.item() * len(mid_price_return)
            acc = torch.where((((output > 0) & (mid_price_return > 0)) |
                                ((output < 0) & (mid_price_return < 0)) |
                                (mid_price_return == 0)
                                ), True, False)
            acc_num += torch.sum(acc).item()
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
                LOB = sample[0][0].to(device, non_blocking=True)
                mark = sample[0][1].to(device, non_blocking=True)
                mid_price_return = sample[1].to(device, non_blocking=True)
                output = model(x_enc=LOB, x_mark_enc=mark[:, :100], x_dec=LOB, x_mark_dec=mark[:, -100:])
                # output = model(x_enc=LOB, x_mark_enc=mark[:, :100], x_dec=LOB[:, -85:], x_mark_dec=mark[:, -100:])
                loss = loss_fn(output, mid_price_return)
                total_loss += loss.item() * len(mid_price_return)
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

        torch.save(model, "../model/FEDformer/FEDformer-{:d}.pt".format(epoch_ndx))

        if min(sum_ks_stat_list) < sum_ks_stat and min(sum_ad_stat_list) < sum_ad_stat:
            no_improved += 1
            if no_improved > stop_threshold:
                break
        else:
            no_improved = 0

        

        # checkpoint = {
        #     'epoch_ndx': epoch_ndx,
        #     'model': model.state_dict(),
        #     'optimizer': optimizer.state_dict()
        # }
        # torch.save(checkpoint, "../model/Checkpoint_FEDformer.pth".format(epoch_ndx))


if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    main(100)
