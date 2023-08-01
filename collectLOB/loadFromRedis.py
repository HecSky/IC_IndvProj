import pickle

import redis
import json

pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
r = redis.Redis(connection_pool=pool)

snapshots = r.zrange("book-BINANCE-BTC-TUSD", -36000, -1)
# snapshots = r.zrange("book-BINANCE-BTC-TUSD", 3, 1_800_003)

V = "9"

timestamp_list = []
PV_list = []
OF_list = []
PVOF_list = []
mid_prices_list = []

counter = 0
for index, snapshot in enumerate(snapshots):
    if index % 100000 == 0:
        print(index)
    snapshot_dict = json.loads(snapshot)
    timestamp = snapshot_dict["timestamp"]
    # print(timestamp)
    book = snapshot_dict["book"]
    asks = book["ask"]
    bids = book["bid"]
    if len(asks) != 10 or len(bids) != 10:
        counter += 1
        continue
    asks_price = []
    asks_volume = []
    bids_price = []
    bids_volume = []
    for ask in asks.items():
        asks_price.append(float(ask[0]))
        asks_volume.append(ask[1])
    for bid in bids.items():
        bids_price.append(float(bid[0]))
        bids_volume.append(bid[1])
    pv_list = []
    of_list = []
    pvof_list = []
    if index == 0:
        for i in range(0, 10):
            pv_list.append(asks_price[i])
            pv_list.append(asks_volume[i])
            pv_list.append(bids_price[i])
            pv_list.append(bids_volume[i])
    else:
        for i in range(0, 10):
            pv_list.append(asks_price[i])
            pv_list.append(asks_volume[i])
            pv_list.append(bids_price[i])
            pv_list.append(bids_volume[i])

            aof = 0.0
            pre_ask_price = PV_list[index - 1 - counter][i * 4]
            if asks_price[i] > pre_ask_price:
                aof = -asks_volume[i]
            elif asks_price[i] == pre_ask_price:
                aof = asks_volume[i] - PV_list[index - 1 - counter][i * 4 + 1]
            else:
                aof = asks_volume[i]

            bof = 0.0
            pre_bid_price = PV_list[index - 1 - counter][i * 4 + 2]
            if bids_price[i] > pre_bid_price:
                bof = bids_volume[i]
            elif bids_price[i] == pre_bid_price:
                bof = bids_volume[i] - PV_list[index - 1 - counter][i * 4 + 3]
            else:
                bof = -bids_volume[i]

            of_list.append(aof)
            of_list.append(bof)

            pvof_list.append(asks_price[i])
            pvof_list.append(asks_volume[i])
            pvof_list.append(aof)
            pvof_list.append(bids_price[i])
            pvof_list.append(bids_volume[i])
            pvof_list.append(bof)

    timestamp_list.append(timestamp)
    PV_list.append(pv_list)
    OF_list.append(of_list)
    PVOF_list.append(pvof_list)
    mid_prices_list.append((asks_price[0] + bids_price[0]) / 2)

f = open('V' + V + '/timestamp.pkl', 'wb')
pickle.dump(timestamp_list[1:], f)
f.close()

# f = open('V' + V + '/PV.pkl', 'wb')
# pickle.dump(PV_list[1:], f)
# f.close()
#
# f = open('V' + V + '/OF.pkl', 'wb')
# pickle.dump(OF_list[1:], f)
# f.close()

f = open('V' + V + '/PVOF.pkl', 'wb')
pickle.dump(PVOF_list[1:], f)
f.close()

f = open('V' + V + '/mid_prices.pkl', 'wb')
pickle.dump(mid_prices_list[1:], f)
f.close()
