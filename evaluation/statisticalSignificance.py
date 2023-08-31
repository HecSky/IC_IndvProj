import pickle
import seaborn as sns
import scipy.stats as ss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.weightstats as ws


def calculate_statistics():
    for model_kind in ["deep", "xgb"]:
        fig, ax = plt.subplots(2, 6, figsize=(18, 10))
        if model_kind == "deep":
            model_names = ["QL", "DQN", "PPO"]
        else:
            model_names = ["XGBoost", "DQN", "PPO"]
        print(model_kind, model_names)

        for market_kind_idx, market_kind in enumerate(["SNL", "NL"]):
            print(market_kind)
            f = open("eval_" + model_kind + "_" + market_kind + "_action_rewards_list.pkl", "rb")
            action_rewards_list = pickle.load(f)
            f.close()

            for dataset_idx, dataset_res in enumerate(action_rewards_list):
                print("Dataset", dataset_idx + 1)
                # min_size = min(len(dataset_res[0]), min(len(dataset_res[1]), len(dataset_res[2])))
                # dataset_model1 = np.compress(dataset_res[0] != 0, dataset_res[0])
                # dataset_model2 = np.compress(dataset_res[1] != 0, dataset_res[1])
                # dataset_model3 = np.compress(dataset_res[2] != 0, dataset_res[2])
                # datasets = [dataset_model1, dataset_model2, dataset_model3]

                datasets = [dataset_res[0], dataset_res[1], dataset_res[2]]
                res = ss.levene(datasets[0], datasets[1], datasets[2])
                print(res)

                series1 = pd.Series(datasets[0], name=model_names[0])
                series2 = pd.Series(datasets[1], name=model_names[1])
                series3 = pd.Series(datasets[2], name=model_names[2])
                df = pd.concat([series1, series2, series3], axis=1)
                df.columns = model_names
                sns.violinplot(data=df, ax=ax[market_kind_idx][dataset_idx], scale='width')
                ax[market_kind_idx][dataset_idx].set_title("Dataset" + str(dataset_idx + 1))

                pvalues_list = []
                for row in range(3):
                    pvalues = []
                    for col in range(3):
                        compareMeans = ws.CompareMeans(ws.DescrStatsW(datasets[row]), ws.DescrStatsW(datasets[col]))
                        stat, pvalue = compareMeans.ztest_ind(usevar='unequal')
                        pvalues.append(pvalue)
                    pvalues_list.append(pvalues)
                print(pvalues_list)

            print("All datasets")
            datasets = []
            for i in range(3):
                tmp = np.concatenate([action_rewards_list[0][i], action_rewards_list[1][i], action_rewards_list[2][i],
                                      action_rewards_list[3][i], action_rewards_list[4][i]])
                datasets.append(tmp)
            res = ss.levene(datasets[0], datasets[1], datasets[2])
            print(res)

            series1 = pd.Series(datasets[0], name=model_names[0])
            series2 = pd.Series(datasets[1], name=model_names[1])
            series3 = pd.Series(datasets[2], name=model_names[2])
            df = pd.concat([series1, series2, series3], axis=1)
            df.columns = model_names
            sns.violinplot(data=df, ax=ax[market_kind_idx][-1], scale="width")
            ax[market_kind_idx][-1].set_title("All Dataset")

            pvalues_list = []
            for row in range(3):
                pvalues = []
                for col in range(3):
                    compareMeans = ws.CompareMeans(ws.DescrStatsW(datasets[row]), ws.DescrStatsW(datasets[col]))
                    stat, pvalue = compareMeans.ztest_ind(usevar="unequal")
                    pvalues.append(pvalue)
                pvalues_list.append(pvalues)
            print(pvalues_list)

        plt.subplots_adjust(wspace=0.7)
        plt.savefig("./statistics_" + model_kind + "_" + market_kind + ".png", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    if True:
        calculate_statistics()
