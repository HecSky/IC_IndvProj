from cryptofeed import FeedHandler
from cryptofeed.backends.redis import BookRedis
from cryptofeed.defines import L2_BOOK
from cryptofeed.exchanges import Binance


def main():
    config = {'log': {'filename': 'redis-demo.log', 'level': 'INFO'}, 'backend_multiprocessing': True}
    f = FeedHandler(config=config)

    f.add_feed(Binance(max_depth=10,
                       symbols=['BTC-TUSD'],
                       channels=[L2_BOOK],
                       callbacks={L2_BOOK: BookRedis(snapshots_only=True)}))

    f.run()


if __name__ == '__main__':
    main()
