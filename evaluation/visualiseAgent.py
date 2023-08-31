import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualise_ql():
    fig, axs = plt.subplots(nrows=3, ncols=9, figsize=(20, 12))

    f = open("ql_actions.pkl", "rb")
    ql_actions_all_side = pickle.load(f)
    f.close()

    side = 0

    for ql_actions in ql_actions_all_side:
        statistic = [np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(6),
                     np.zeros(3)]
        for row in ql_actions:
            for i in range(9):
                statistic[i][row[i]] += 1

        for i in range(9):
            statistic[i] = (statistic[i] - min(statistic[i])) / len(ql_actions)
            print(statistic[i])
            if i <= 6:
                ax = sns.barplot(x=[0, 1, 2, 3, 4], y=statistic[i], ax=axs[side][i])
                ax.set_title("RoR" + str(i))
            elif i == 7:
                ax = sns.barplot(x=[0, 1, 2, 3, 4, 5], y=statistic[i], ax=axs[side][i])
                ax.set_title("CoP")
            elif i == 8:
                ax = sns.barplot(x=[0, 1, 2], y=statistic[i], ax=axs[side][i])
                ax.set_title("Side")

        side += 1

    plt.subplots_adjust(wspace=0.7)
    plt.savefig('./visual_ql.png', bbox_inches='tight')
    plt.show()

def visualise_dqn():
    fig, axs = plt.subplots(nrows=3, ncols=9, figsize=(20, 12))

    f = open("dqn_actions.pkl", "rb")
    ql_actions_all_side = pickle.load(f)
    f.close()

    side = 0

    for ql_actions in ql_actions_all_side:
        statistic = [np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5),
                     np.zeros(3)]
        for row in ql_actions:
            for i in range(9):
                statistic[i][row[i]] += 1

        for i in range(9):
            statistic[i] = (statistic[i] - min(statistic[i])) / len(ql_actions)
            print(statistic[i])
            if i <= 6:
                ax = sns.barplot(x=[0, 1, 2, 3, 4], y=statistic[i], ax=axs[side][i])
                ax.set_title("RoR" + str(i))
            elif i == 7:
                ax = sns.barplot(x=[0, 1, 2, 3, 4], y=statistic[i], ax=axs[side][i])
                ax.set_title("CoP")
            elif i == 8:
                ax = sns.barplot(x=[0, 1, 2], y=statistic[i], ax=axs[side][i])
                ax.set_title("Side")

        side += 1

    plt.subplots_adjust(wspace=0.7)
    plt.savefig('./dqn_actions.png', bbox_inches='tight')
    plt.show()

def visualise_ppo():
    fig, axs = plt.subplots(nrows=3, ncols=9, figsize=(20, 12))

    f = open("ppo_actions.pkl", "rb")
    ql_actions_all_side = pickle.load(f)
    f.close()

    side = 0

    for ql_actions in ql_actions_all_side:
        statistic = [np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5),
                     np.zeros(3)]
        for row in ql_actions:
            for i in range(9):
                statistic[i][row[i]] += 1

        for i in range(9):
            statistic[i] = (statistic[i] - min(statistic[i])) / len(ql_actions)
            print(statistic[i])
            if i <= 6:
                ax = sns.barplot(x=[0, 1, 2, 3, 4], y=statistic[i], ax=axs[side][i])
                ax.set_title("RoR" + str(i))
            elif i == 7:
                ax = sns.barplot(x=[0, 1, 2, 3, 4], y=statistic[i], ax=axs[side][i])
                ax.set_title("CoP")
            elif i == 8:
                ax = sns.barplot(x=[0, 1, 2], y=statistic[i], ax=axs[side][i])
                ax.set_title("Side")

        side += 1

    plt.subplots_adjust(wspace=0.7)
    plt.savefig('./ppo_actions.png', bbox_inches='tight')
    plt.show()

def visualise_xgb():
    fig, axs = plt.subplots(nrows=2, ncols=7, figsize=(20, 12))

    f = open("xgb_actions.pkl", "rb")
    ql_actions_all_side = pickle.load(f)[:2]
    f.close()

    side = 0

    for ql_actions in ql_actions_all_side:
        statistic = [np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)]
        for row in ql_actions:
            for i in range(7):
                statistic[i][row[i]] += 1

        for i in range(7):
            statistic[i] = (statistic[i] - min(statistic[i])) / len(ql_actions)
            print(statistic[i])
            ax = sns.barplot(x=[0, 1, 2, 3, 4], y=statistic[i], ax=axs[side][i])
            ax.set_title("RoR" + str(i))

        side += 1

    plt.subplots_adjust(wspace=0.7)
    plt.savefig('./xgb_actions.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    if True:
        visualise_ql()
        visualise_dqn()
        visualise_ppo()
        visualise_xgb()