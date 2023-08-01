import pickle
from datetime import datetime

from matplotlib import pyplot as plt


def draw_mid_price(V: str):
    f = open("./V" + V + "/mid_prices.pkl", 'rb')
    mid_price = pickle.load(f)
    f.close()

    f = open("./V" + V + "/timestamp.pkl", 'rb')
    timestamps = pickle.load(f)
    f.close()
    datetime_list = list(map(datetime.fromtimestamp, timestamps))

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.16, top=0.92)
    plt.plot(datetime_list, mid_price)
    plt.xlabel('Date')
    plt.ylabel('BTC-TUSD')
    plt.title('Mid price')
    plt.xticks(rotation=30)
    plt.grid(True)

    # Show the chart
    plt.savefig("../figure/" + V + ".png")
    plt.show()


if __name__ == "__main__":
    for i in range(1, 10):
        draw_mid_price(str(i))
