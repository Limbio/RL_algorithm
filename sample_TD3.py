import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import torch
import matplotlib.pyplot as plt


device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(input_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, output_dim)

        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.softmax(self.l3(x), dim=-1) # Use softmax to output probabilities
        return x

class Critic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(input_dim + output_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = F.relu(self.l1(torch.cat([state, action], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class TD3Agent:
    def __init__(self, cfg):
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.max_action = cfg.num_tasks

        self.actor = Actor(self.input_dim, self.output_dim, self.max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), cfg.lr_actor)

        self.critic_1 = Critic(self.input_dim, self.output_dim).to(device)
        self.critic_2 = Critic(self.input_dim, self.output_dim).to(device)
        self.critic_target_1 = copy.deepcopy(self.critic_1)
        self.critic_target_2 = copy.deepcopy(self.critic_2)
        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), cfg.lr_critic)
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), cfg.lr_critic)

        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.policy_noise = cfg.policy_noise
        self.noise_clip = cfg.noise_clip
        self.policy_freq = cfg.policy_freq
        self.total_it = 0

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)

        with torch.no_grad():
            action_probs = self.actor(state_tensor).cpu().squeeze().numpy()

        # 确定在当前状态下哪些任务是可用的
        available_tasks = [i-1 for i in range(self.output_dim+1) if i not in state]

        # 根据这些任务调整行动的概率
        adjusted_probs = [action_probs[i] if i in available_tasks else 0 for i in range(self.output_dim)]
        adjusted_probs /= np.sum(adjusted_probs)

        # 选择一个行动
        action = np.random.choice(self.output_dim, p=adjusted_probs)
        return action

    def train(self, replay_buffer, batch_size):
        self.total_it += 1

        # Sample from buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        # print(action.shape)  # <--- Add this line

        state = torch.tensor(state, dtype=torch.float).to(device)
        action = F.one_hot(torch.LongTensor(action), self.output_dim).float().to(device)
        # print(action.shape)  # <--- And this line

        reward = torch.tensor(reward, dtype=torch.float).to(device).unsqueeze(1)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        done = torch.tensor(done, dtype=torch.float).to(device).unsqueeze(1)

        # Compute targets
        with torch.no_grad():
            noise = (torch.randn(next_state.size(0), self.output_dim) * self.policy_noise).to(device).clamp(
                -self.noise_clip, self.noise_clip)
            next_action_probs = self.actor_target(next_state)
            next_action = torch.multinomial(next_action_probs, 1).squeeze(1)
            next_action = F.one_hot(next_action, self.output_dim).float()

            target_Q1 = self.critic_target_1(next_state, next_action)
            target_Q2 = self.critic_target_2(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Update critics
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)
        loss_Q1 = F.mse_loss(current_Q1, target_Q)
        loss_Q2 = F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer_1.zero_grad()
        loss_Q1.backward()
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.zero_grad()
        loss_Q2.backward()
        self.critic_optimizer_2.step()

        # Update actor and target networks
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def train_td3(cfg, agent, env, replay_buffer):
    print("开始训练TD3！")

    rewards = []
    total_reward = 0
    prev_avg_reward = -float('inf')
    no_change_count = 0
    avg_rewards_every_100 = []  # 用于存储每100轮的平均奖励


    for episode in range(cfg.episodes):
        state = env.reset()
        done = False
        # print(episode)
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward = env.total_profit
            env.index *= cfg.eps_decay  # Assuming you want to include the decay like in DDPG

            if len(replay_buffer) >= cfg.batch_size:
                agent.train(replay_buffer, cfg.batch_size)

        rewards.append(total_reward)
        env.total_profit = 0  # reset total_reward for the next episode
        avg_reward = np.mean(rewards[-cfg.patience:])

        if episode % 100 == 0:
            avg_rewards_every_100.append(np.mean(rewards[-100:]))
            print(f"第 {episode} 轮, 平均奖励: {np.mean(rewards[-100:])}, 平均奖励 (最近 {cfg.patience} 轮): {avg_reward}")

        # Early stopping condition based on the average reward's change
        if episode >= cfg.patience and avg_reward == prev_avg_reward:
            no_change_count += 1
            if no_change_count >= cfg.max_episodes:
                print(f"提前停止训练。平均奖励连续 {cfg.max_episodes} 轮没有改善。")
                break
        else:
            no_change_count = 0
        prev_avg_reward = avg_reward

    plt.plot([i*100 for i in range(len(avg_rewards_every_100))], avg_rewards_every_100,label='TD3')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward (Every Episodes)')
    plt.title('Average Reward Over Time')
    plt.legend(loc='upper right')

    plt.savefig("DQN_plot.png")
    return np.mean(rewards)


class ReplayBuffer:
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def __len__(self):
        return len(self.storage)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in ind:
            s, a, r, s_, d = self.storage[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_)
            dones.append(d)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

def test_td3(cfg, agent, env):
    """Test the trained TD3 agent over multiple episodes and return the average reward and optimal allocation."""
    total_rewards = []
    optimal_allocations = []

    for episode in range(cfg.test_episodes):
        state = env.reset()
        optimal_actions = []
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
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
