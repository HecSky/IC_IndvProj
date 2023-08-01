import pickle
import torch

f = open('mid_prices.pkl', 'rb')
mid_prices = pickle.load(f)
f.close()

count_list = []
same = False
counter = 1
previous_mid_price = 0.0
for mid_price in mid_prices:
    if previous_mid_price == mid_price:
        counter += 1
    else:
        count_list.append(counter)
        previous_mid_price = mid_price
        counter = 1

print(sum(count_list)/len(count_list))

# mid_prices = torch.tensor(mid_prices)[:int(len(mid_prices)*0.8)]
# MP_std_mean = torch.std_mean(mid_prices, dim=0, keepdim=True)
# MP_std = MP_std_mean[0]
# MP_mean = MP_std_mean[1]

# with open("../zscore_parameters", "r") as f:
#     lines = f.read()
#     lines = lines.split("\n")[4:]
#     parameters = []
#     for line in lines:
#         parameters.append(torch.tensor(float(line)))
#     MP_std = parameters[0]
#     MP_mean = parameters[1]

# mid_prices = (mid_prices - MP_mean) / MP_std
# q = torch.tensor([0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1])
# print(torch.quantile(mid_prices, q))