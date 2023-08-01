import pickle
from collections import OrderedDict

import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, layer_sizes: list[int], dropout: float):
        super(Actor, self).__init__()
        self.activation = nn.PReLU()
        self.Softmax = nn.Softmax(dim=1)
        self.Dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)])

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            if idx != (len(self.layers) - 1):
                x = self.Dropout(self.activation(layer(x)))
            else:
                x = layer(x)
        x = self.Softmax(x)
        return x


class Critic(nn.Module):
    def __init__(self, layer_sizes: list[int], dropout: float):
        super(Critic, self).__init__()
        self.activation = nn.PReLU()
        self.Dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)])

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            if idx != (len(self.layers) - 1):
                x = self.Dropout(self.activation(layer(x)))
            else:
                x = layer(x)
        return x

class PPOAgent(nn.Module):
    def __init__(self,
                 network: list,
                 model_V: str,
                 model_type: str,
                 num_factors: int = 9,
                 lr_actor: float = 2e-4,
                 lr_critic: float = 2e-4,
                 update_times: int = 3,
                 clip_rate: float = 0.25,
                 gamma: float = 0.96,
                 dropout: float = 0.01,
                 batch_size: int = 8192,
                 entropy_coefficient: float = 0.05,
                 ):
        super(PPOAgent, self).__init__()
        self.device_name = "cpu"
        self.device = torch.device(self.device_name)
        self.actor = Actor(network, dropout=dropout).to(self.device)
        self.critic = Critic(network, dropout=dropout).to(self.device)
        # self.MSE_Loss = nn.MSELoss()
        self.batch_size = batch_size
        self.num_factors = num_factors
        self.update_times = update_times
        self.clip_rate = clip_rate
        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient

        self.optimizer_actor = torch.optim.AdamW(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.AdamW(self.critic.parameters(), lr=lr_critic)
        lambda_lr_actor = lambda epoch: max(0.98 ** epoch, 2e-1)
        lambda_lr_critic = lambda epoch: max(0.98 ** epoch, 2e-1)
        self.scheduler_actor = torch.optim.lr_scheduler.LambdaLR(self.optimizer_actor, lr_lambda=lambda_lr_actor)
        self.scheduler_critic = torch.optim.lr_scheduler.LambdaLR(self.optimizer_critic, lr_lambda=lambda_lr_critic)

        self.idx_trajectory = 0
        self.observations = torch.empty((self.batch_size, self.num_factors), dtype=torch.float, device=self.device)
        self.rewards = torch.empty((self.batch_size, 1), dtype=torch.float, device=self.device)
        self.actions = torch.empty((self.batch_size, 1), dtype=torch.long, device=self.device)
        self.log_probabilities = torch.empty((self.batch_size, 1), dtype=torch.float, device=self.device)
        self.reward_to_go = None

        f = open('../parameters/normalise_paras_V' + model_V + "_" + model_type + '.pkl', 'rb')
        normalise_paras = pickle.load(f)
        f.close()
        self.ror_mean = normalise_paras[2].item()
        self.ror_std = normalise_paras[3].item()

    def reset(self, counter: int, next_batch: bool, next_dataset: bool, initial_reset: bool):
        self.idx_trajectory = 0
        self.observations = torch.empty((self.batch_size, self.num_factors), dtype=torch.float, device=self.device)
        self.rewards = torch.empty((self.batch_size, 1), dtype=torch.float, device=self.device)
        self.actions = torch.empty((self.batch_size, 1), dtype=torch.long, device=self.device)
        self.log_probabilities = torch.empty((self.batch_size, 1), dtype=torch.float, device=self.device)
        self.reward_to_go = None

    def predict_train(self, observation):
        direction_probabilities = self.actor(observation.unsqueeze(0)).detach()[0]
        # nan_mask = torch.isnan(direction_probabilities)
        # direction_probabilities[nan_mask] = 0.0
        direction_probabilities = direction_probabilities + 0.01
        action = torch.multinomial(direction_probabilities, 1)
        log_probabilities = direction_probabilities[action].log()
        self.observations[self.idx_trajectory] = observation
        self.actions[self.idx_trajectory] = action
        self.log_probabilities[self.idx_trajectory] = log_probabilities
        return action

    def predict(self, observation):
        with torch.no_grad():
            direction_probabilities = self.actor(observation.unsqueeze(0)).detach()[0]
            direction = torch.argmax(direction_probabilities).to(self.device)
        return direction

    def record_reward(self, reward):
        self.rewards[self.idx_trajectory] = reward
        self.idx_trajectory += 1

    def calculate_reward_to_go(self):
        length_trajectory = len(self.rewards)
        self.reward_to_go = torch.zeros((length_trajectory, 1), dtype=torch.float, device=self.device)
        R = 0.0
        for i in range(length_trajectory - 1, -1, -1):
            R = self.rewards[i][0].item() + self.gamma * R
            self.reward_to_go[i][0] = R
        self.reward_to_go = (self.reward_to_go - torch.mean(self.reward_to_go)) / (torch.std(self.reward_to_go) + 1e-7)

    def calculate_advantages(self):
        values = self.critic(self.observations).detach()
        values = (values - torch.mean(values)) / (torch.std(values) + 1e-7)
        advantages = self.reward_to_go - values
        advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-7)
        return advantages

    def calculate_log_probabilities_numerator(self):
        direction_probabilities = self.actor(self.observations)
        direction_probabilities = torch.gather(direction_probabilities, dim=1, index=self.actions) + 0.01
        log_probabilities = direction_probabilities.log()
        return log_probabilities

    def actor_loss(self, advantages):
        log_probabilities_cur = self.calculate_log_probabilities_numerator()
        ratio_of_probabilities = torch.exp(log_probabilities_cur - self.log_probabilities)

        actor_loss_first = advantages * ratio_of_probabilities

        actor_loss_second = advantages * torch.clamp(ratio_of_probabilities, 1.0 - self.clip_rate, 1.0 + self.clip_rate)

        actor_loss = -torch.minimum(actor_loss_first, actor_loss_second).mean()

        entropy_loss = -torch.mean(-log_probabilities_cur)
        return actor_loss + self.entropy_coefficient * entropy_loss

    def critic_loss(self):
        values = self.critic(self.observations)
        values = torch.gather(values, dim=1, index=self.actions)
        # critic_loss = self.MSE_Loss(values, self.reward_to_go)
        critic_loss = ((values - self.reward_to_go) ** 2).mean()
        return critic_loss

    def cut_tensor(self, next_dataset: bool):
        if next_dataset:
            self.observations = self.observations[:self.idx_trajectory]
            self.rewards = self.rewards[:self.idx_trajectory]
            self.actions = self.actions[:self.idx_trajectory]
            self.log_probabilities = self.log_probabilities[:self.idx_trajectory]

    def update(self, counter: int, next_batch: bool, next_dataset: bool, initial_update: bool):
        self.cut_tensor(next_dataset=next_dataset)
        self.calculate_reward_to_go()
        advantages = self.calculate_advantages()
        for i in range(self.update_times):
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            actor_loss = self.actor_loss(advantages)
            critic_loss = self.critic_loss()
            actor_loss.backward()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1, norm_type=2)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1, norm_type=2)
            self.optimizer_actor.step()
            self.optimizer_critic.step()
        self.scheduler_actor.step()
        self.scheduler_critic.step()
        self.reset(counter=counter, next_batch=next_batch, next_dataset=next_dataset, initial_reset=initial_update)
