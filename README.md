# Combining deep learning on order books with reinforcement learning for profitable trading



## Description

A framework for high-frequency trading, and this is the code of the implementation of the project.


## Code guide
### Run on Linux
Because this project is developed on IDE PyCharm on Windows 10, if Python scripts run on Linux, you may face some errors. To solve, you can try to add following code at the beginning of relevant code files.
```python
import os
import sys
sys.path.append(os.path.abspath(".."))
```

### Data
- Data is shared on Google Drive (https://drive.google.com/drive/folders/12YsNFIgeGM7QbK8XfUMCZWEy1zAXCddM?usp=sharing) because of the limitation of GitHub.
- Collect data: start Redis server and run "/collectLOB/collectLOBdata.py"
- Save data to local: create new folder in "collectLOB" with "V[number]", then run "/collectLOB/loadFromRedis.py" (remember to modify code at line 12 to corresponding [number])

### Train time series forecasting models
- Create new folder in folder "model". The name of the new folder should be the name of the model
- Modify "V" and "para_name" of LOBDataset. V is dataset name and para_name is parameter name
- Run scripts in folder "alpha" with name "train[model name]Extractor_multi.py"

### Generate parameters and alphas
- Move the target model to folder "model" with name: [Model name]-V[Datasets].pt
- Modify "V", "model_type" in "/alpha/getQuantile_multi.py" and specify the model in specific epoch by modifying line 15 and 16
- Run "/alpha/getQuantile_multi.py" to generate parameters
- New folder in folder "alphas" with file name: [Model name]-V[Datasets]
- Modify "model_type", "model_V", "data_V", "DQN" to generate alphas for dataset in "data_V" by using specific model. If alphas is for deep reinforcement learning model or XGBoost, "DQN" should be True.

### Train reinforcement learning agent
- Q-Learning: set parameters in "/RL/QLearning_multi.py" and run it. Then a model will be generated as "/RL/V[Dataset]-[Model name].pkl".
- DQN: set parameters in "/RL/DQN.py" and run it. Then a model will be generated as "/RL/V[Dataset]-[Model name]-DQN.pt".
- PPO: set parameters in "/RL/DQN.py" and run it. Then a model will be generated as "/RL/V[Dataset]-[Model name]-PPO.pt".
- XGBoost: set parameters in "/RL/xgb.py" and run it. Then a model will be generated as "/RL/bst".

### Evaluation
- Evaluate Q-Learning PnL performance with different time series forecasting models: run "/RL/eval_ql_agents.py".
- Evaluate reinforcement learning PnL performance: run "/RL/eval_deep_agents.py".
- Evaluate deep reinforcement learning PnL performance with XGBoost: run "/RL/eval_xgb_agents.py".
- To control "short", modify code in function "test" in "DQN", "PPO", "QLearning_multi" and "xgb". There is code: "if action == 0: action = 1". If allow short, then comment it. If not, uncomment it.
- "/evaluation/latexTable.py" is used to generate table in LaTeX.
- “/evaluation/visualiseAgent.py” visualise how agent makes decisions.
- “/evaluation/statisticalSignificance.py" calcualte the statistical significance for reward per trade of different models.

### Figures
- Figures are saved in folder "figure".

## Code citation
### Time series forecasting Models
https://github.com/cure-lab/LTSF-Linear

@inproceedings{Zeng2022AreTE,
  title={Are Transformers Effective for Time Series Forecasting?},
  author={Ailing Zeng and Muxi Chen and Lei Zhang and Qiang Xu},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}

These codes are in the folder "alpha", and there are some modifications.

### Data collection
https://github.com/bmoscon/cryptofeed

This GitHub repository is created by Bryant Moscon(bmoscon@gmail.com). It is used to collect limit order book data.

It is used in "/collectLOB/draw_ror_distribution.py"

### DQN
The file "/RL/DQNutils.py" is from coursework material "Reinforcement Learning 2022 Coursework Material" in Department of Computing in Imperial College London.

### XGBoost
https://github.com/dmlc/xgboost

https://xgboost.readthedocs.io/en/stable/index.html

The XGBoost used in "/RL/xgb.py" is from the library "xgboost" on Github.