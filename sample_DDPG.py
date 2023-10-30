import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch
import random
import matplotlib.pyplot as plt


device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Actor网络
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)


# Critic网络
class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.fc(x)


class DDPGAgent:
    def __init__(self, cfg):
        self.actor = Actor(cfg.input_dim, cfg.output_dim).to(device)
        self.critic = Critic(cfg.input_dim, cfg.output_dim).to(device)

        self.target_actor = Actor(cfg.input_dim, cfg.output_dim).to(device)
        self.target_critic = Critic(cfg.input_dim, cfg.output_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        self.gamma = cfg.gamma
        self.memory = []
        self.batch_size = cfg.batch_size
        self.memory_size = cfg.memory_size
        self.loss_fn = nn.MSELoss()
        self.is_training = cfg.is_training
        self.num_agents = cfg.input_dim
        self.num_tasks = cfg.output_dim
        self.tau = 0.995  # soft update coefficient
        self.action_dim = cfg.output_dim

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
        action_probs = action_probs.squeeze().cpu().numpy()
        available_tasks = [i-1 for i in range(self.num_tasks+1) if i not in state]
        action_probs = [action_probs[i] if i in available_tasks else 0 for i in range(self.num_tasks)]
        action_probs /= np.sum(action_probs)
        action = np.random.choice(np.arange(self.num_tasks), p=action_probs)
        return action,action_probs

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(device)

        actions_one_hot = F.one_hot(torch.LongTensor(actions), self.action_dim).float().to(device)

        # actions = torch.LongTensor(actions).unsqueeze(-1)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)

        # Critic update
        next_actions = self.target_actor(next_states)
        q_target_next = self.target_critic(next_states, next_actions)
        q_target = rewards + self.gamma * q_target_next.squeeze()
        q_current = self.critic(states, actions_one_hot)
        critic_loss = self.loss_fn(q_current, q_target.unsqueeze(-1))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)


def train_ddpg(cfg, agent, env):
    print("开始训练DDPG！")
    rewards = []
    total_reward = 0
    prev_avg_reward = -float('inf')
    no_change_count = 0
    avg_rewards_every_100 = []  # 用于存储每100轮的平均奖励


    for episode in range(cfg.episodes):
        # print("episode:",episode)
        state = env.reset()
        done = False

        while not done:
            action,_ = agent.choose_action(state)
            next_state, reward, done, _, = env.step(action)
            agent.store_transition(state, action, reward, next_state)
            total_reward = env.total_profit
            agent.update()
            state = next_state
            env.index *= cfg.eps_decay
            # print("not done")

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

    plt.plot([i*100 for i in range(len(avg_rewards_every_100))], avg_rewards_every_100,label='DDPG')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward (Every Episodes)')
    plt.title('Average Reward Over Time')
    plt.legend(loc='upper right')
    plt.savefig("DQN_plot.png")

    return np.mean(rewards)


def test_ddpg(cfg, agent, env):
    """Test the trained DDPG agent over multiple episodes and return the average reward and optimal allocation."""
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
