import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from sklearn.model_selection import train_test_split

# 读取Movielens数据集
movies = pd.read_csv("tensorflow/data/ml-latest-small/movies.csv")
ratings = pd.read_csv("tensorflow/data/ml-latest-small/ratings.csv")

# 合并电影信息和评分数据
ratings = ratings.merge(movies, on='movieId')

# 数据预处理
ratings['userId'] = ratings['userId'].astype(str)
ratings['movieId'] = ratings['movieId'].astype(str)

# 划分训练集和测试集
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# 定义状态表示
state_size = 5  # 用户最近5个电影评分作为状态

def create_state_sequence(data):
    state_sequence = data.groupby('userId').apply(lambda x: x.sort_values('timestamp').tail(state_size))
    state_sequence = state_sequence[['userId', 'movieId', 'rating']]
    return state_sequence.reset_index(drop=True)

train_state_sequence = create_state_sequence(train_data)

# 定义动作空间
action_space = ratings['movieId'].unique()

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x

# 定义强化学习代理
class ReinforcementLearningAgent:
    def __init__(self, state_size, action_size, alpha, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子

        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=alpha)
        self.criterion = nn.CrossEntropyLoss()

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        action_probs = self.policy_network(state_tensor)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    def update(self, state, action, reward, next_state):
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)

        action_probs = self.policy_network(state_tensor)
        action_log_probs = torch.log(action_probs)
        selected_action_log_prob = action_log_probs[action]

        next_action_probs = self.policy_network(next_state_tensor)
        next_state_value = torch.max(next_action_probs)

        target = reward + self.gamma * next_state_value

        loss = -selected_action_log_prob * target

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# 创建强化学习代理
agent = ReinforcementLearningAgent(state_size, len(action_space), alpha=0.001, gamma=0.9)

# 定义训练过程
def train():
    num_episodes = 1000  # 训练轮数
    max_steps = 100  # 每轮最大步数

    for episode in range(num_episodes):
        state_sequence = train_state_sequence.sample(frac=1).reset_index(drop=True)  # 打乱顺序

        for i in range(len(state_sequence)):
            if i + state_size >= len(state_sequence):
                break

            state = state_sequence.loc[i:i+state_size-1, 'rating'].values
            action = agent.select_action(state)

            next_state = state_sequence.loc[i+1:i+state_size, 'rating'].values
            reward = state_sequence.loc[i+state_size, 'rating']

            agent.update(state, action, reward, next_state)

# 训练强化学习代理
train()

# 使用训练好的策略网络进行推荐
def recommend(state):
    state_tensor = torch.FloatTensor(state)
    action_probs = agent.policy_network(state_tensor)
    action = torch.argmax(action_probs).item()
    recommended_movie = action_space[action]
    return recommended_movie

# 示例：生成推荐
test_users = test_data['userId'].unique()
for user in test_users[:5]:
    user_state = train_state_sequence[train_state_sequence['userId'] == user].tail(state_size)['rating'].values
    recommendation = recommend(user_state)
    print(f"User {user} Recommendation: {recommendation}")
