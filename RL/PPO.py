import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from RL.PPOutils import PPOAgent
from backtest.backtestSystem_train_preprocess import LOBEnv_train_preprocess
from backtest.backtestSystem_valid_preprocess import LOBEnv_valid_preprocess


class PPO_agent(object):
    def __init__(self, agent: PPOAgent
                 ) -> None:
        self.agent = agent
        
    def solve(self, env, episode_num, model_type):
        rewards = []
        for episode in range(1, episode_num + 1):
            print("{:d}/{:d}".format(episode, episode_num))
            if episode % 100 == 0:
                torch.save(self.agent.actor, "V1;2;3;4-" + model_type + "-PPO.pt")
    
            state, done = env.reset()
    
            if self.agent.device_name == "cpu":
                state_ = state[0][:]
                state_.append((state[1] - self.agent.ror_mean) / self.agent.ror_std)
                state_.append(state[2])
                state = torch.tensor(state_)
            else:
                state = torch.cat(
                    (state[0], torch.tensor([((state[1] - self.agent.ror_mean) / self.agent.ror_std), state[2]]).to(self.agent.device)),
                    dim=0)
    
            total_reward = 0.0
            positive_reward = 0.0
            abs_total_reward = 0.0

            long_num = 0
            mid_num = 0
            short_num = 0
            counter = 0
            while not done:
                if env.trade_index % 100000 == 0:
                    print("Episode{:d}:{:d}".format(episode, int(env.trade_index)))

                action = self.agent.predict_train(state).item()
    
                if action == 0:
                    short_num += 1
                elif action == 1:
                    mid_num += 1
                else:
                    long_num += 1
    
                previous_dataset_index = env.dataset_index
                next_state, reward, done = env.step(action)
                self.agent.record_reward(reward)
    
                if self.agent.device_name == "cpu":
                    state_ = next_state[0][:]
                    state_.append((next_state[1] - self.agent.ror_mean) / self.agent.ror_std)
                    state_.append(next_state[2])
                    state = torch.tensor(state_)
                else:
                    state = torch.cat((next_state[0], torch.tensor(
                        [((next_state[1] - self.agent.ror_mean) / self.agent.ror_std), next_state[2]]).to(self.agent.device)), dim=0)
    
                total_reward += reward
                abs_total_reward += abs(reward)
                if reward > 0:
                    positive_reward += reward
    
                counter += 1

                # condition1 = counter == int(agent.batch_size * (4/5)) and agent.idx_trajectory == agent.batch_size
                condition1 = counter == self.agent.batch_size and self.agent.idx_trajectory == self.agent.batch_size
                condition2 = env.dataset_index != previous_dataset_index
                condition3 = counter == self.agent.batch_size
                if condition1 or condition2 or condition3:
                    self.agent.update(counter, condition1, condition2, condition3)
                    counter = 0
    
            rewards.append(total_reward)
    
            print("Train{:d}-Reward:{:.8f}-Accuracy:{:.8f}".format(episode, total_reward,
                                                                   positive_reward / abs_total_reward))
            print("Long:{:d}, Mid:{:d}, Short:{:d}".format(long_num, mid_num, short_num))
            print("Average_reward:{:.8f}".format(sum(rewards) / len(rewards)))
    
    def test(self, env, model_V: str = "", data_V: str = "", model_type: str = "", return_res: bool = False):
        with torch.no_grad():
            self.agent.actor.eval()
            state, done = env.reset()
    
            if self.agent.device_name == "cpu":
                state_ = state[0][:]
                state_.append((state[1] - self.agent.ror_mean) / self.agent.ror_std)
                state_.append(state[2])
                state = torch.tensor(state_)
            else:
                state = torch.cat(
                    (state[0], torch.tensor([((state[1] - self.agent.ror_mean) / self.agent.ror_std), state[2]]).to(self.agent.device)),
                    dim=0)
    
            total_reward = 0.0
            abs_total_reward = 0.0
            positive_reward = 0.0
            last_action = 1
            action_duration = 1
            action_durations = []
            action_rewards = []
            action_reward = 0
            rewards = [0]
            actions = []
            trade_num = 0

            long_num = 0
            mid_num = 0
            short_num = 0
            while not done:
                if env.index % 100000 == 0:
                    print("Index:{:d}".format(int(env.index)))
    
                action = self.agent.predict(state).item()
    
                # if action == 0:
                #     action = 1
    
                actions.append(action)
    
                if action == 0:
                    short_num += 1
                elif action == 1:
                    mid_num += 1
                else:
                    long_num += 1
    
                next_state, reward, done = env.step(action)
    
                if self.agent.device_name == "cpu":
                    state_ = next_state[0][:]
                    state_.append((next_state[1] - self.agent.ror_mean) / self.agent.ror_std)
                    state_.append(next_state[2])
                    state = torch.tensor(state_)
                else:
                    state = torch.cat((next_state[0], torch.tensor(
                        [((next_state[1] - self.agent.ror_mean) / self.agent.ror_std), next_state[2]]).to(self.agent.device)), dim=0)
    
                total_reward += reward
                abs_total_reward += abs(reward)
                rewards.append(total_reward)
    
                if last_action == action:
                    action_duration += 1
                    action_reward += reward
                else:
                    action_durations.append(action_duration)
                    action_duration = 1
                    last_action = action
                    trade_num += 1
                    action_rewards.append(action_reward)
                    action_reward = reward
    
                if reward > 0:
                    positive_reward += reward
    
            if return_res:
                return rewards, trade_num, np.array(action_durations).mean().item(), action_rewards
            else:
                rewards = np.array(rewards)
                xs = np.arange(0, len(rewards))
                color = []
                for i in range(0, len(rewards) - 1):
                    if actions[i] == 0:
                        color.append("#008000")
                    elif actions[i] == 1:
                        color.append("#0000FF")
                    elif actions[i] == 2:
                        color.append("#FF0000")
                    else:
                        color.append("#FFFFFF")
    
                points = np.array([xs, rewards]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                line_collection = LineCollection(segments, color=color)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.set_xlim(min(xs) - (max(xs) - min(xs)) / 20, max(xs) + (max(xs) - min(xs)) / 20)
                ax.set_ylim(min(rewards) - (max(rewards) - min(rewards)) / 20,
                            max(rewards) + (max(rewards) - min(rewards)) / 20)
                ax.add_collection(line_collection)
                plt.ylabel("PnL")
                plt.xlabel("Time")
                plt.savefig("../figure/" + model_type + "-V" + model_V + "_" + data_V + "_PPO.png")
                plt.show()
    
                print("Valid-Reward:{:.8f}-Accuracy:{:.8f}".format(total_reward, positive_reward / abs_total_reward))
                print("Long:{:d}, Mid:{:d}, Short:{:d}".format(long_num, mid_num, short_num))
                action_durations = np.array(action_durations)
                print("Average side duration:{:8f}".format(action_durations.mean()))


if __name__ == "__main__":
    torch.set_printoptions(precision=8)

    alpha_model_type = "DLinear"
    model_V = "1;2;3;4"
    ppo_Agent = PPOAgent(network=[9, 64, 64, 64, 3], model_V=model_V, model_type=alpha_model_type)
    
    ppo_agent = PPO_agent(ppo_Agent)

    env_train = LOBEnv_train_preprocess(data_V="1;2;3;4", model_V=model_V, model_type=alpha_model_type, deep=True)
    ppo_agent.solve(env_train, 200, alpha_model_type)

    ppo_agent.agent.actor = torch.load("V" + model_V + "-" + alpha_model_type + "-PPO.pt")

    data_Vs = ["5", "6", "7", "8", "9"]
    # data_Vs = ["9"]
    for d_V in data_Vs:
        env_valid = LOBEnv_valid_preprocess(data_V=d_V, model_V=model_V, model_type=alpha_model_type, deep=True)
        ppo_agent.test(env_valid, data_V=d_V, model_V=model_V, model_type=alpha_model_type)
