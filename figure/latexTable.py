import pickle

def generate_latex_table(data):
    rows = len(data)
    cols = len(data[0])

    latex_code = "\\begin{tabular}{|" + "c|" * cols + "}\n"
    latex_code += "\\hline\n"

    for row in data:
        latex_code += " & ".join(str(cell) for cell in row)
        latex_code += " \\\\\n"
        latex_code += "\\hline\n"

    latex_code += "\\end{tabular}"

    return latex_code

f = open("eval_ql_SNL_res.pkl", 'rb')
# f = open("eval_deep_NL_res.pkl", 'rb')
# f = open("eval_xgb_NL_res.pkl", 'rb')
res = pickle.load(f)
f.close()

for i in range(5):
    template = [
        ["Model", "Total reward", "Trade", "Avg duration", "Avg reward", "Std reward"],
        ["DLinear", 8, 9, 10, 11, 12],
        ["LSTM", 14, 15, 16, 17, 18],
        ["Encoder", 20, 21, 22, 23, 24],
        ["CNNLSTM", 26, 27, 28, 29, 30],
        ["InformerW", 32, 33, 34, 35, 36],
        ["AutoformerW", 38, 39, 40, 41, 42],
        ["FEDformerW", 44, 45, 46, 47, 48]
    ]
    # template = [
    #     ["Model", "Total reward", "Trade", "Avg duration", "Avg reward", "Std reward"],
    #     ["QLearning", 8, 9, 10, 11, 12],
    #     ["DQN", 14, 15, 16, 17, 18],
    #     ["PPO", 20, 21, 22, 23, 24],
    # ]
    # template = [
    #     ["Model", "Total reward", "Trade", "Avg duration", "Avg reward", "Std reward"],
    #     ["XGBoost", 8, 9, 10, 11, 12],
    #     ["DQN", 14, 15, 16, 17, 18],
    #     ["PPO", 20, 21, 22, 23, 24],
    # ]
    for j in range(5):
        for k in range(7):
    # for j in range(5):
    #     for k in range(3):
            if j == 0:
                value = str(round(res[j][i][k], 2))
            elif j == 1:
                value = str(res[j][i][k])
            elif j == 2:
                value = str(round(res[j][i][k], 2))
            elif j == 3:
                value = str(round(res[j][i][k], 5))
            elif j == 4:
                value = str(round(res[j][i][k], 5))

            template[k+1][j+1] = value
    latex_table = generate_latex_table(template)
    print(i)
    print(latex_table)