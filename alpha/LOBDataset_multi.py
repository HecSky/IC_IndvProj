import pickle
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import Dataset
from alpha.utils.timefeatures import time_features


class LOBDataset(Dataset):
    def __init__(self, V: str = "1", dataset_type: str = "train", para_name: str = "1", ts: bool= False):
        super().__init__()

        self.device = torch.device("cuda")
        self.ts = ts

        self.LOB_list = []
        self.mid_price_list = []
        self.marks_list = []
        LOB_tensor = None

        Vs = V.split(";")

        self.num_dataset = 0
        for i, v in enumerate(Vs):
            f = open("../collectLOB/V" + v + "/PVOF.pkl", "rb")
            LOB = pickle.load(f)
            LOB = torch.tensor(LOB)
            f.close()

            f = open("../collectLOB/V" + v + "/mid_prices.pkl", "rb")
            mid_price = pickle.load(f)
            mid_price = torch.tensor(mid_price)
            f.close()

            if self.ts:
                f = open("../collectLOB/V" + v + "/timestamp.pkl", "rb")
                timestamps = pickle.load(f)
                f.close()
                datetime_list = list(map(datetime.fromtimestamp, timestamps))
                datetime_index = pd.DatetimeIndex(datetime_list)
                marks = time_features(datetime_index, "ms")
                marks = torch.tensor(marks, dtype=torch.float)
                marks = torch.permute(marks, dims=(1, 0))
                self.marks_list.append(marks)

            self.LOB_list.append(LOB)
            self.mid_price_list.append(mid_price)

            self.num_dataset += 1

            if i == 0:
                LOB_tensor = torch.clone(LOB)
            else:
                LOB_tensor = torch.cat((LOB_tensor, LOB), dim=0)

        price_indices = torch.tensor(range(0, 58, 3))
        volume_indices = torch.tensor(range(1, 59, 3))
        OF_indices = torch.tensor(range(2, 60, 3))

        if dataset_type == "train":
            with open("../parameters/zscore_parameters_PVOF_" + para_name, "w") as f:
                price_std_mean = torch.std_mean(LOB_tensor[:, price_indices].reshape(-1), dim=0, keepdim=True)
                volume_std_mean = torch.std_mean(LOB_tensor[:, volume_indices].reshape(-1), dim=0, keepdim=True)
                OF_std_mean = torch.std_mean(LOB_tensor[:, OF_indices].reshape(-1), dim=0, keepdim=True)
                price_std = price_std_mean[0]
                price_mean = price_std_mean[1]
                volume_std = volume_std_mean[0]
                volume_mean = volume_std_mean[1]
                OF_std = OF_std_mean[0]
                OF_mean = OF_std_mean[1]
                f.write(str(price_std.item()) + "\n")
                f.write(str(price_mean.item()) + "\n")
                f.write(str(volume_std.item()) + "\n")
                f.write(str(volume_mean.item()) + "\n")
                f.write(str(OF_std.item()) + "\n")
                f.write(str(OF_mean.item()))

        elif dataset_type == "valid":
            with open("../parameters/zscore_parameters_PVOF_" + para_name, "r") as f:
                lines = f.read()
                lines = lines.split("\n")
                parameters = []
                for line in lines:
                    parameters.append(torch.tensor(float(line)))
                price_std = parameters[0]
                price_mean = parameters[1]
                volume_std = parameters[2]
                volume_mean = parameters[3]
                OF_std = parameters[4]
                OF_mean = parameters[5]

        self.future_mid_price_index = []
        for i in range(2, 15, 2):
            self.future_mid_price_index.append(i)
        self.future_mid_price_index = torch.tensor(self.future_mid_price_index)

        self.total_length = 0
        self.linear_lengths = [0]
        for LOB in self.LOB_list:
            LOB[:, price_indices] = (LOB[:, price_indices] - price_mean) / price_std
            LOB[:, volume_indices] = (LOB[:, volume_indices] - volume_mean) / volume_std
            LOB[:, OF_indices] = (LOB[:, OF_indices] - OF_mean) / OF_std
            if self.ts:
                self.total_length += len(LOB) - (99 + self.future_mid_price_index[-1]) - 1
                # self.total_length += len(LOB) - (99 + self.future_mid_price_index[-1]) - 15
            else:
                self.total_length += len(LOB) - (99 + self.future_mid_price_index[-1])
            linear_length = 0
            linear_length += self.total_length
            self.linear_lengths.append(linear_length)


    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        counter = 0
        for i in range(1, len(self.linear_lengths)+1):
            linear_length = self.linear_lengths[i]
            if idx < linear_length:
                break
            else:
                counter += 1
        real_idx = idx - self.linear_lengths[counter]
        # print(idx, real_idx, counter, self.linear_lengths[counter])
        # print(real_idx + 99 + self.future_mid_price_index[-1], len(self.mid_price_list[counter]))
        data = self.LOB_list[counter][real_idx:real_idx + 100]
        label = (self.mid_price_list[counter][real_idx + 99 + self.future_mid_price_index]
                 - self.mid_price_list[counter][real_idx + 99 + 1]) / self.mid_price_list[counter][real_idx + 99 + 1]
        if self.ts:
            mark = self.marks_list[counter][real_idx:real_idx + 101]
            # mark = self.marks_list[counter][real_idx:real_idx + 115]
            return (data, mark), label
        else:
            return data, label
