import pickle

import torch
from torch.utils.data import DataLoader

from alpha.LOBDataset_multi import LOBDataset


def main(model_type: str = "LSTM", model_V: str = "1", data_V: str = "1", ts: bool = False, DQN: bool = False):
    device = torch.device("cuda")
    model = torch.load("../model/" + model_type + "-V" + model_V + ".pt").to(device)
    model.eval()

    Vs = data_V.split(";")

    f = open('../parameters/normalise_paras_V' + model_V + "_" + model_type + '.pkl', 'rb')
    normalise_paras = pickle.load(f)
    f.close()
    mean = torch.tensor(normalise_paras[0], dtype=torch.float, device=device)
    std = torch.tensor(normalise_paras[1], dtype=torch.float, device=device)

    for V in Vs:
        dataset_train = LOBDataset(V=V, dataset_type="valid", para_name=model_V, ts=ts)
        dataloader_train = DataLoader(dataset_train, batch_size=10240, num_workers=8)

        outputs = None

        with torch.no_grad():
            model.eval()
            for batch_ndx, sample in enumerate(dataloader_train):
                if timestamp:
                    LOB = sample[0][0].to(device, non_blocking=True)
                    mark = sample[0][1].to(device, non_blocking=True)
                    output = model(x_enc=LOB, x_mark_enc=mark[:, :100], x_dec=LOB, x_mark_dec=mark[:, -100:])
                else:
                    LOB = sample[0].to(device, non_blocking=True)
                    output = model(LOB)

                if DQN:
                    output = (output - mean) / std

                if outputs == None:
                    outputs = output
                else:
                    outputs = torch.cat((outputs, output), dim=0)

        outputs_list = outputs.tolist()

        if DQN:
            f = open(model_type + "-V" + model_V + "-DQN/V" + V + ".pkl", 'wb')
            pickle.dump(outputs_list[:], f)
            f.close()
        else:
            f = open(model_type + "-V" + model_V + "/V" + V + ".pkl", 'wb')
            pickle.dump(outputs_list[:], f)
            f.close()


if __name__ == "__main__":
    with torch.no_grad():
        timestamp = False
        main(model_type="DLinear", model_V="1;2;3;4", data_V="1;2;3;4;5;6;7;8;9", ts=timestamp, DQN=False)
