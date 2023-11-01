import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import random
import math

# Define the DQN architecture
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.skip_connection = nn.Identity()

    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1) + self.skip_connection(x1))
        return self.fc3(x2)

# Define the Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

    def push(self, *args):
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define the DQL algorithm
class DQL:
    def __init__(self, input_size, output_size, gamma=0.99, lr=0.001):
        self.policy_net = DQN(input_size, output_size)
        self.target_net = DQN(input_size, output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(10000)
        self.gamma = gamma
        self.batch_size = 64
        self.update_factor = 0.995

    def select_action(self, state, eps_threshold):
        if np.random.rand() < eps_threshold:
            return torch.tensor([[random.randrange(state.shape[1])]])
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = self.memory.transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_net(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.update_factor * target_param.data + (1.0 - self.update_factor) * policy_param.data)

# ... Code for initializing environment and training loop here ...

def train_dql(dql, env, num_episodes, M, N, eps_start=0.9, eps_end=0.05, eps_decay=0.995):
    """
    dql: DQL对象
    env: 环境对象，例如OpenAI Gym环境
    num_episodes: 训练的总周期数
    M: 导弹的数量
    N: 每个导弹的目标数量
    eps_start: ε-贪婪策略的起始ε值
    eps_end: ε-贪婪策略的结束ε值
    eps_decay: 每次ε衰减的值
    """
    global_reward_list = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for i in range(M):
            state_mi = env.observe_missile_state(i)
            action_space_mi = []

            for j in range(N):
                action_vector_aj = env.describe_target(i, j)
                action_space_mi.append(action_vector_aj)

            # ε-贪婪策略选择行动
            eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * episode / eps_decay)
            action = dql.select_action(torch.tensor([state_mi], dtype=torch.float32), eps_threshold)

            # 执行选择的行动，获取奖励和下一个状态
            next_state, local_reward, done, _ = env.step(action.item())
            global_reward = ...  # 这里你需要一个公式或方法来计算全局奖励
            reward = α * local_reward + (1 - α) * global_reward

            # 存储转换到回放记忆中
            if done:
                next_state = None
            dql.memory.push(torch.tensor([state_mi], dtype=torch.float32), action, torch.tensor([reward], dtype=torch.float32), torch.tensor([next_state], dtype=torch.float32))

            # 移到下一个状态
            state_mi = next_state

            # 优化模型
            dql.optimize_model()

            total_reward += reward

            # 判断是否结束
            if done:
                break

        # 更新目标网络
        dql.update_target_net()

        global_reward_list.append(total_reward)
        print(f"Episode {episode + 1}: Total reward received: {total_reward}")

    return global_reward_list




# 使用示例
env = ...  # 使用你的环境初始化，例如OpenAI Gym的某个环境
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
dql_agent = DQL(input_size, output_size)

# 开始训练
train_dql(dql_agent, env, num_episodes=1000)


