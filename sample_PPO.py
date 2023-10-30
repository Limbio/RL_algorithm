import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch
import random
import matplotlib.pyplot as plt



device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        self.to(device)


    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)


class PPOAgent:
    def __init__(self, cfg):
        self.policy_net = PPOPolicy(cfg.input_dim, cfg.output_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon if cfg.is_training else 0.0
        self.memory = []
        self.batch_size = cfg.batch_size
        self.memory_size = cfg.memory_size
        self.is_training = cfg.is_training
        self.num_agents = cfg.input_dim
        self.num_tasks = cfg.output_dim
        self.clip_epsilon = 0.2

    def choose_action(self, state):
        available_tasks = [i-1 for i in range(self.num_tasks + 1) if i not in state]

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = self.policy_net(state_tensor)
        distribution = torch.distributions.Categorical(probs)
        action = distribution.sample().item()
        if action in available_tasks:
            return action
        else:
            return random.choice(available_tasks)  # fallback to random action if sampled action is not available

    def update(self, old_probs, states, actions, rewards):
        if len(self.memory) < self.batch_size:
            return
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        # print(states.device)
        # print(actions.device)
        # print(rewards.device)

        # Calculate new action probabilities
        probs = self.policy_net(states)
        distribution = torch.distributions.Categorical(probs)
        new_probs = distribution.log_prob(actions)

        # Calculate the ratio
        old_probs = torch.stack(old_probs).squeeze().to(device)
        # print(old_probs.device)
        ratio = (new_probs - old_probs).exp()

        # Calculate surrogate losses
        surrogate1 = ratio * rewards
        surrogate2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * rewards
        loss = -torch.min(surrogate1, surrogate2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store_transition(self, state, action, reward, old_prob):
        self.memory.append((state, action, reward, old_prob))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)


def train_ppo(cfg, agent, env):
    print("PPO开始训练！")

    rewards = []
    prev_avg_reward = -float('inf')
    no_change_count = 0
    total_reward = 0
    avg_rewards_every_100 = []  # 用于存储每100轮的平均奖励


    for episode in range(cfg.episodes):
        state = env.reset()
        done = False
        old_probs, states, actions, rewards_batch = [], [], [], []

        while not done:
            action = agent.choose_action(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            # print(state_tensor.device)
            probs = agent.policy_net(state_tensor)
            distribution = torch.distributions.Categorical(probs)
            # old_prob = distribution.log_prob(torch.tensor([action])).item()
            # old_prob = distribution.log_prob(torch.tensor([action])).to(device)

            action_tensor = torch.tensor([action]).to(device)
            old_prob = distribution.log_prob(action_tensor)

            old_probs.append(old_prob)

            next_state, reward, done, _ = env.step(action)
            total_reward = env.total_profit
            old_probs.append(old_prob)
            states.append(state)
            actions.append(action)
            rewards_batch.append(reward)
            env.index *= cfg.eps_decay
            state = next_state

        agent.update(old_probs, states, actions, rewards_batch)
        rewards.append(total_reward)
        env.total_profit = 0
        avg_reward = np.mean(rewards[-cfg.patience:])

        if episode % 100 == 0:
            avg_rewards_every_100.append(np.mean(rewards[-100:]))
            print(f"第 {episode} 轮, 平均奖励: {np.mean(rewards[-100:])}, 平均奖励 (最近 {cfg.patience} 轮): {avg_reward}")

        # 如果连续patience轮的平均奖励没有变化，就提前停止训练
        if episode >= cfg.patience and avg_reward == prev_avg_reward:
            no_change_count += 1
            if no_change_count >= cfg.max_episodes:
                print(f"提前停止训练。平均奖励连续 {cfg.max_episodes} 轮没有改善。")
                break
        else:
            no_change_count = 0
        prev_avg_reward = avg_reward

    plt.plot([i*100 for i in range(len(avg_rewards_every_100))], avg_rewards_every_100,label='PPO')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward (Every 100 Episodes)')
    plt.title('Average Reward Over Time')
    plt.legend(loc='upper right')

    plt.savefig("DQN_plot.png")
    return np.mean(rewards)

def test_ppo(cfg,agent, env,):
    """Test the trained PPO agent over multiple episodes and return the average reward and optimal allocation."""
    total_rewards = []
    optimal_allocations = []

    for episode in range(cfg.test_episodes):
        state = env.reset()
        optimal_actions = []
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            optimal_actions.append(action)
            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)
        optimal_allocations.append(optimal_actions)
        print(f"Episode {episode + 1}, Reward: {total_reward}, Optimal Allocation: {optimal_actions}")

    avg_reward = np.mean(total_rewards)
    print(f"平均奖励 (共 {cfg.test_episodes} 轮): {avg_reward}")

    return avg_reward, optimal_allocations