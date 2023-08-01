import pickle
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

from RL.QLearning_multi import QL_agent
from backtest.backtestSystem_valid_preprocess import LOBEnv_valid_preprocess


def to_str(dt):
    return dt.strftime("%m-%d-%H")


def eval_ql(dataset: str, model_list: list, training_datasets: str):
    f = open("../collectLOB/V" + dataset + "/timestamp.pkl", 'rb')
    timestamps = pickle.load(f)
    f.close()
    datetime_list = list(map(datetime.fromtimestamp, timestamps))
    datetime_list = list(map(to_str, datetime_list))[100:]
    total_rewards_list = []
    trade_num_list = []
    average_action_duration_list = []
    average_reward_list = []
    std_reward_list = []
    for model_name in model_list:
        ql_agent = QL_agent(model_type=model_name, V=training_datasets)
        f = open("V1;2;3;4-" + model_name + ".pkl", 'rb')
        ql_agent.policy = pickle.load(f)
        f.close()
        env_test = LOBEnv_valid_preprocess(data_V=dataset, model_V=training_datasets, model_type=model_name)
        rewards, trade_num, average_action_duration, action_rewards = ql_agent.test(env_test, return_res=True)
        total_rewards_list.append(rewards)
        trade_num_list.append(trade_num)
        average_action_duration_list.append(average_action_duration)
        action_rewards = np.array(action_rewards)
        average_reward = action_rewards.mean()
        average_reward_list.append(average_reward)
        std_reward = action_rewards.std()
        std_reward_list.append(std_reward)

    tmp = []
    fig, ax = plt.subplots(figsize=(8.09, 5))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.16, top=0.92)
    for i, model_name in enumerate(model_list):
        total_reward = total_rewards_list[i]
        tmp.append(total_reward[-1])
        x_values = np.arange(len(total_reward))
        plt.plot(x_values, total_reward, label=model_name)
        # plt.plot(datetime_list[:min_length], total_reward[:min_length], label=model_name)
    date_length = len(datetime_list)
    x_index = [int(i / 10) for i in range(0, date_length, int(date_length / 8))]
    x_label = [datetime_list[i] for i in range(0, date_length, int(date_length / 8))]
    plt.xticks(x_index, x_label)
    plt.legend(loc="upper left")
    plt.xlabel('Date')
    plt.ylabel('PnL')
    plt.xticks(rotation=30)
    plt.grid(True)
    plt.savefig("../figure/eval_ql_SNL_" + dataset + ".png")
    plt.show()
    total_rewards_list = tmp
    return total_rewards_list, trade_num_list, average_action_duration_list, average_reward_list, std_reward_list


if __name__ == "__main__":
    if True:
        list_model = ["DLinear", "LSTM", "Encoder", "CNNLSTM", "InformerW", "AutoformerW", "FEDformerW"]
        model_training_datasets = "1;2;3;4"
        testing_datasets = ["5", "6", "7", "8", "9"]
        total_rewards_list = []
        trade_num_list = []
        average_action_duration_list = []
        average_reward_list = []
        std_reward_list = []
        for testing_dataset in testing_datasets:
            total_rewards, trade_num, average_action_duration, average_reward, std_reward = eval_ql(
                dataset=testing_dataset,
                model_list=list_model,
                training_datasets=model_training_datasets)
            total_rewards_list.append(total_rewards)
            trade_num_list.append(trade_num)
            average_action_duration_list.append(average_action_duration)
            average_reward_list.append(average_reward)
            std_reward_list.append(std_reward)
        result = [total_rewards_list, trade_num_list, average_action_duration_list, average_reward_list,
                  std_reward_list]
        print(result)
        f = open("../figure/eval_ql_SNL_res.pkl", "wb")
        pickle.dump(result, f)
        f.close()