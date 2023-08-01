import pickle
from numpy.random import default_rng

Vs = ["1", "2", "3", "4"]

# rng = default_rng()

total = 0.0

for V in Vs:
    # f = open("./V" + V + "/mid_prices.pkl", 'rb')
    # mid_price = pickle.load(f)
    # f.close()
    # mid_price_length = len(mid_price)
    # vals = rng.standard_normal(mid_price_length)
    #
    # print(len(vals))
    # print(vals.mean(), vals.std())
    # f = open('./V' + V + '/mid_prices_test.pkl', 'wb')
    # pickle.dump(vals.tolist(), f)
    # f.close()

    f = open("./V" + V + "/mid_prices_test.pkl", 'rb')
    mid_price = pickle.load(f)
    f.close()
    for i in range(len(mid_price)):
        total += mid_price[i]

print(total)