import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cpu")


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        self.to(device)  # 将模型移动到设备

    def forward(self, x):
        return self.fc(x)


class DQNAgent:
    def __init__(self, cfg):
        self.q_net = DQN(cfg.input_dim, cfg.output_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.01)
        self.gamma = cfg.gamma
        self.epsilon_start = 1.0  # 初始epsilon值，表示完全随机探索
        self.epsilon_final = 0.01  # 最终epsilon值，表示最小的随机探索
        self.epsilon_decay = 2000  # 衰减率
        self.epsilon = self.epsilon_start  # 当前epsilon值        self.memory = []
        self.batch_size = cfg.batch_size
        self.memory_size = cfg.memory_size
        self.loss_fn = nn.MSELoss()
        self.is_training = cfg.is_training
        self.num_agents = cfg.input_dim
        self.num_tasks = cfg.output_dim
        self.target_q_net = DQN(cfg.input_dim, cfg.output_dim).to(device)
        self.update_target_freq = 100
        self.update_count = 0
        self.sync_q_net()
        self.memory = []
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        self.eps_decay = cfg.eps_decay

    def sync_q_net(self):  # 同步主Q网络到目标Q网络
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def update_epsilon(self, episode_num):
        # 使用指数衰减计算epsilon
        self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1. * episode_num / self.epsilon_decay)

    def choose_action(self, state):
        available_tasks = [i - 1 for i in range(self.num_tasks + 1) if i not in state]

        self.epsilon *= self.eps_decay if self.is_training else 1.0
        if self.is_training and np.random.rand() < self.epsilon:
            return random.choice(available_tasks)  # 只从未分配的任务中选择

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        sorted_actions = q_values.argsort(descending=True)
        for action in sorted_actions[0]:
            if action.item() in available_tasks:
                return action.item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(-1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)

        curr_q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values

        loss = self.loss_fn(curr_q_values, target_q_values.unsqueeze(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 定期从主Q网络更新目标Q网络
        self.update_count += 1
        if self.update_count % self.update_target_freq == 0:
            self.sync_q_net()

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)


class DDQNAgent(DQNAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(-1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)

        curr_q_values = self.q_net(states).gather(1, actions)

        # 使用主Q网络选择下一个动作
        _, best_actions = self.q_net(next_states).max(1, keepdim=True)

        # 使用目标Q网络得到这个动作的Q值
        next_q_values = self.target_q_net(next_states).gather(1, best_actions)

        target_q_values = rewards + self.gamma * next_q_values

        loss = self.loss_fn(curr_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新学习率调度器
        self.lr_scheduler.step()

        # 定期从主Q网络更新目标Q网络
        self.update_count += 1
        if self.update_count % self.update_target_freq == 0:
            self.sync_q_net()


def train_dqn(cfg, agent, env):
    print("DQN开始训练！")
    rewards = []
    total_reward = 0
    prev_avg_reward = -float('inf')
    no_change_count = 0
    avg_rewards_every_100 = []  # 用于存储每100轮的平均奖励

    for episode in range(cfg.episodes):
        state = env.reset()
        agent.update_epsilon(episode)  # 更新 epsilon 值
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, = env.step(action)
            agent.store_transition(state, action, reward, next_state)
            total_reward = env.total_profit
            agent.update()
            state = next_state
            env.index *= cfg.eps_decay

        rewards.append(total_reward)
        env.total_profit = 0
        avg_reward = np.mean(rewards[-cfg.patience:])

        if episode % 100 == 0:
            avg_rewards_every_100.append(np.mean(rewards[-100:]))
            print(
                f"第 {episode} 轮, 平均奖励: {np.mean(rewards[-100:])}, 平均奖励 (最近 {cfg.patience} 轮): {avg_reward}")

        # 如果连续patience轮的平均奖励没有变化，就提前停止训练
        if episode >= cfg.patience and avg_reward == prev_avg_reward:
            no_change_count += 1
            if no_change_count >= cfg.max_episodes:
                print(f"提前停止训练。平均奖励连续 {cfg.max_episodes} 轮没有改善。")
                break
        else:
            no_change_count = 0
        prev_avg_reward = avg_reward

    plt.plot([i * 100 for i in range(len(avg_rewards_every_100))], avg_rewards_every_100, label='DQN')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward (Every 100 Episodes)')
    plt.title('Average Reward Over Time')
    plt.legend(loc='upper right')
    plt.savefig("DQN_plot.png")
    # plt.show()

    return np.mean(rewards)


def test_dqn(cfg, agent, env):
    """Test the trained DQN agent over multiple episodes and return the average reward and optimal allocation."""
    total_rewards = []
    optimal_allocations = []
    normalized_rewards = []

    for episode in range(cfg.test_episodes):
        state = env.reset()
        optimal_actions = []
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            optimal_actions.append(action)
            total_reward = env.total_profit
            state = next_state

        total_possible_reward = sum(env.task_rewards.flatten()) - sum(env.agent_costs)
        normalized_reward = total_reward / total_possible_reward  # 计算标准化奖励

        env.total_profit = 0

        total_rewards.append(total_reward)
        normalized_rewards.append(normalized_reward)
        optimal_allocations.append(optimal_actions)
        print(f"Episode {episode + 1}, Reward: {total_reward}, Optimal Allocation: {optimal_actions}")

    avg_reward = np.mean(total_rewards)
    avg_normalized_reward = np.mean(normalized_rewards)  # 计算所有episode的平均标准化奖励
    print(f"平均奖励 (共 {cfg.test_episodes} 轮): {avg_reward}")
    print("标准化平均奖励: ", avg_normalized_reward)

    return avg_reward, optimal_allocations
