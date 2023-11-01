import numpy as np
import torch
import optuna
import os
import matplotlib.pyplot as plt
from itertools import permutations
from sample_DQN import DQNAgent, train_dqn, test_dqn,DDQNAgent
from sample_PPO import PPOAgent, train_ppo, test_ppo
from sample_DDPG import DDPGAgent, train_ddpg, test_ddpg
from sample_TD3 import TD3Agent, train_td3, test_td3, ReplayBuffer


# 1. Configuration
class BaseConfig:
    def __init__(self, N_task, N_agent, is_training):
        # Basic configuration
        self.num_agents = N_agent
        self.num_tasks = N_task
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_training = is_training
        self.max_episodes = 3000
        self.num_epochs = 1
        self.patience = 3000
        self.episodes = 100000
        self.test_episodes = 1000

        # Paths for saving and loading models
        self.model_path = "/Users/limbo/PycharmProjects/wta_rl/saved_model"

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_path)

    def load_model(self, model):
        model.load_state_dict(torch.load(self.model_path))
        model.eval()


class DQNConfig(BaseConfig):
    def __init__(self, N_task, N_agent, is_training):
        super().__init__(N_task, N_agent, is_training)
        self.model_path = "/Users/limbo/PycharmProjects/wta_rl/saved_model/dqn_model.pth"
        self.input_dim = self.num_agents
        self.output_dim = self.num_tasks
        self.epsilon = 0.93
        self.gamma = 0.83
        self.batch_size = 32
        self.memory_size = 3500
        self.eps_decay = 0.955
        # Add other DQN specific parameters if needed

class DDQNConfig(BaseConfig):
    def __init__(self, N_task, N_agent, is_training):
        super().__init__(N_task, N_agent, is_training)
        self.model_path = "/Users/limbo/PycharmProjects/wta_rl/saved_model/ddqn_model.pth"
        self.input_dim = self.num_agents
        self.output_dim = self.num_tasks
        self.epsilon = 0.93
        self.gamma = 0.83
        self.batch_size = 32
        self.memory_size = 3500
        self.eps_decay = 0.955
        # Add other DQN specific parameters if needed
# Similarly add configurations for PPO, DDPG, and TD3
# ... [previous code]

class PPOConfig(BaseConfig):
    def __init__(self, N_task, N_agent, is_training):
        super().__init__(N_task, N_agent, is_training)
        self.model_path = "/Users/limbo/PycharmProjects/wta_rl/saved_model/ppo_model.pth"
        self.input_dim = self.num_agents
        self.output_dim = self.num_tasks
        self.lr = 0.0003
        self.betas = (0.9, 0.999)
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.policy_clip = 0.1
        # Add other PPO specific parameters if needed


class DDPGConfig(BaseConfig):
    def __init__(self, N_task, N_agent, is_training):
        super().__init__(N_task, N_agent, is_training)
        self.model_path = "/Users/limbo/PycharmProjects/wta_rl/saved_model/ddpg_model.pth"
        self.actor_input_dim = self.num_agents
        self.actor_output_dim = self.num_tasks
        self.critic_input_dim = self.num_agents + self.num_tasks
        self.lr_actor = 0.0001
        self.lr_critic = 0.001
        self.gamma = 0.99
        self.tau = 0.005
        self.noise_std = 0.2
        # Add other DDPG specific parameters if needed


class TD3Config(BaseConfig):
    def __init__(self, N_task, N_agent, is_training):
        super().__init__(N_task, N_agent, is_training)
        self.model_path_actor = "/Users/limbo/PycharmProjects/wta_rl/saved_model/td3_actor.pth"
        self.model_path_critic = "/Users/limbo/PycharmProjects/wta_rl/saved_model/td3_critic.pth"
        self.input_dim = self.num_agents
        self.output_dim = self.num_tasks
        self.critic_input_dim = self.num_agents + self.num_tasks  # 注意这里可能需要修改，根据你的Critic网络设计
        self.lr_actor = 0.0003
        self.lr_critic = 0.001
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.replaybuffer_max_size = int(1e5)  # 设为整数，因为这将作为列表的最大大小
        self.batch_size = 100  # 增加批处理大小，这通常是一个超参数
        self.max_action = self.num_tasks
        self.eps_decay = 0.955
        # 如果有更多TD3特定的参数，可以继续添加


# ... [rest of the code]
class AgentFactory:
    def get_agent(self, agent_type, config):
        if agent_type == "DQN":
            return DQNAgent(config)
        elif agent_type == "DDQN":
            return DDQNAgent(config)
        elif agent_type == "PPO":
            return PPOAgent(config)
        elif agent_type == "DDPG":
            return DDPGAgent(config)
        elif agent_type == "TD3":
            return TD3Agent(config)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def get_config(self, agent_type, N_task, N_agent):
        if agent_type == "DQN":
            return DQNConfig(N_task, N_agent)
        elif agent_type == "PPO":
            return PPOConfig(N_task, N_agent)
        elif agent_type == "DDPG":
            return DDPGConfig(N_task, N_agent)
        elif agent_type == "TD3":
            return TD3Config(N_task, N_agent,True)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")


class MultiTaskEnvironment:
    def __init__(self, cfg):
        self.num_agents = cfg.num_agents
        self.num_tasks = cfg.num_tasks
        self.state = np.zeros(cfg.num_agents, dtype=np.int64)
        self.task_rewards, self.agent_costs = self._generate_probs()
        self.assigned_tasks = []
        self.assigned_agents = []
        self.is_training = cfg.is_training
        self.total_profit = 0
        self.index = 1

    def _generate_probs(self):
        if cfg.is_training == 0:
            np.random.seed(0)
        elif cfg.is_training != 1:
            raise ValueError("Invalid mode. Choose either 0 or 1.")

        rewards_agent = np.random.randint(10, 100, size=(self.num_agents, self.num_tasks))
        costs = np.random.randint(10, 50, size=self.num_agents)
        np.random.seed(None)
        return rewards_agent, costs

    def reset(self):
        self.state = np.zeros(self.num_agents, dtype=np.int64)
        self.assigned_tasks = []
        self.assigned_agents = []
        self.total_profit = 0
        # 如果是训练模式，重新生成奖励和代价
        if self.is_training:
            self.task_rewards, self.agent_costs = self._generate_probs()
        return self.state

    def step(self, action):
        total_reward = 0
        reward = 0
        agent_idx = len(self.assigned_agents)

        # 如果这个任务还没有被分配，且还有agent可用
        if action not in self.assigned_tasks and agent_idx < self.num_agents:
            available_agents = [idx for idx in range(self.num_agents) if idx not in self.assigned_agents]
            max_reward_diff = -np.inf
            selected_agent = None

            for agent in available_agents:
                reward_diff = self.task_rewards[agent][action] - self.agent_costs[agent]
                if reward_diff > max_reward_diff:
                    max_reward_diff = reward_diff
                    selected_agent = agent

            if selected_agent is not None:
                self.assigned_tasks.append(action)
                self.assigned_agents.append(selected_agent)
                self.state[selected_agent] = action + 1

            self.total_profit += max_reward_diff

            reward = self.index * max_reward_diff + (1 - self.index) * self.total_profit / len(self.assigned_agents)

        # 为结束条件进行检查
        done = len(self.assigned_tasks) == self.num_tasks or agent_idx == self.num_agents
        # print("len(self.assigned_tasks)",len(self.assigned_tasks))
        # print("self.num_tasks",self.num_tasks)
        # print("len(self.assigned_agents)",len(self.assigned_agents))
        # print("self.num_agents",self.num_agents)
        return self.state, reward, done, {}


# 2. Training and Testing
def train(agent_type, cfg, env):
    best_reward = -float("inf")  # 初始化最佳奖励为负无穷

    if agent_type == "DQN":
        agent = DQNAgent(cfg)
        for epoch in range(cfg.num_epochs):
            avg_reward = train_dqn(cfg, agent, env)
            if avg_reward > best_reward:
                best_reward = avg_reward
                cfg.save_model(agent.q_net)  # Assuming q_net is the model you want to save
    elif agent_type == "DDQN":
        agent = DDQNAgent(cfg)
        for epoch in range(cfg.num_epochs):
            avg_reward = train_dqn(cfg, agent, env)
            if avg_reward > best_reward:
                best_reward = avg_reward
                cfg.save_model(agent.q_net)  # Assuming q_net is the model you want to save
    elif agent_type == "PPO":
        agent = PPOAgent(cfg)
        for epoch in range(cfg.num_epochs):
            avg_reward = train_ppo(cfg, agent, env)
            if avg_reward > best_reward:
                best_reward = avg_reward
                cfg.save_model(agent.policy_net)  # Replace with appropriate model

    elif agent_type == "DDPG":
        agent = DDPGAgent(cfg)
        for epoch in range(cfg.num_epochs):
            avg_reward = train_ddpg(cfg, agent, env)
            if avg_reward > best_reward:
                best_reward = avg_reward
                cfg.save_model(agent.actor)  # Assuming actor is the model you want to save for DDPG

    elif agent_type == "TD3":
        agent = TD3Agent(cfg)
        for epoch in range(cfg.num_epochs):
            avg_reward = train_td3(cfg, agent, env,ReplayBuffer(cfg.replaybuffer_max_size))
            if avg_reward > best_reward:
                best_reward = avg_reward
                cfg.save_model(agent.actor)  # Assuming actor is the model you want to save for TD3

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return best_reward


def test(agent_type, cfg, env):
    if agent_type == "DQN":
        agent = DQNAgent(cfg)
        cfg.load_model(agent.q_net)
        return test_dqn(cfg,agent, env)
    elif agent_type == "PPO":
        agent = PPOAgent(cfg)
        cfg.load_model(agent.policy_net)
        return test_ppo(cfg,agent, env)
    elif agent_type == "DDPG":
        agent = DDPGAgent(cfg)
        cfg.load_model(agent.actor) # Adjusted to actor for DDPG
        return test_ddpg(agent, env)
    elif agent_type == "TD3":
        agent = TD3Agent(cfg)
        cfg.load_model(agent.actor) # Adjusted to actor for TD3
        return test_td3(agent, env)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# 3. Optuna Objective
def objective(trial, agent_type, base_cfg, env):
    factory = AgentFactory()
    cfg = factory.get_config(agent_type, base_cfg.num_tasks, base_cfg.num_agents)

    if agent_type == "DQN":
        cfg.epsilon = trial.suggest_float("epsilon", 0.5, 0.95)
        cfg.gamma = trial.suggest_float("gamma", 0.75, 0.99)
        cfg.batch_size = trial.suggest_int("batch_size", 16, 64)
        cfg.memory_size = trial.suggest_int("memory_size", 1000, 10000)
        cfg.eps_decay = trial.suggest_float("eps_decay", 0.9, 0.99)

    elif agent_type == "PPO":
        cfg.lr = trial.suggest_float("lr", 1e-4, 3e-4, log=True)
        cfg.gamma = trial.suggest_float("gamma", 0.9, 0.99)
        cfg.eps_clip = trial.suggest_float("eps_clip", 0.1, 0.3)
        cfg.policy_clip = trial.suggest_float("policy_clip", 0.05, 0.2)

    elif agent_type == "DDPG":
        cfg.lr_actor = trial.suggest_float("lr_actor", 1e-4, 1e-3, log=True)
        cfg.lr_critic = trial.suggest_float("lr_critic", 1e-4, 1e-3, log=True)
        cfg.gamma = trial.suggest_float("gamma", 0.9, 0.99)
        cfg.tau = trial.suggest_float("tau", 0.001, 0.01)
        cfg.noise_std = trial.suggest_float("noise_std", 0.1, 0.5)

    elif agent_type == "TD3":
        cfg.lr_actor = trial.suggest_float("lr_actor", 1e-4, 1e-3, log=True)
        cfg.lr_critic = trial.suggest_float("lr_critic", 1e-4, 1e-3, log=True)
        cfg.gamma = trial.suggest_float("gamma", 0.9, 0.99)
        cfg.tau = trial.suggest_float("tau", 0.001, 0.01)
        cfg.policy_noise = trial.suggest_float("policy_noise", 0.1, 0.5)
        cfg.noise_clip = trial.suggest_float("noise_clip", 0.1, 1)
        cfg.policy_freq = trial.suggest_int("policy_freq", 1, 4)

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    average_reward = train(agent_type, cfg, env)
    return -average_reward  # Optuna tries to minimize the objective, so return negative reward for maximization


# 4. Visualization
def plot_results(episode_rewards, a, title=None):
    """
    Plot the results.

    Parameters:
    - episode_rewards: List of rewards obtained from the agent during training
    - a: The factor to aggregate the rewards, every b/a episodes a point will be plotted
    - title: Title of the plot (optional)
    """
    # Calculate average rewards for every b/a episodes
    avg_rewards = [sum(episode_rewards[i:i + a]) / a for i in range(0, len(episode_rewards), a)]

    plt.figure(figsize=(10, 5))
    plt.plot(avg_rewards, '-o')
    if title:
        plt.title(title)
    plt.xlabel(f"Every {a} Episodes")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.show()


def plot_combined_results(algorithm_rewards, a):
    """
    Plot the combined results of multiple algorithms on a single graph.

    Parameters:
    - algorithm_rewards: Dictionary with keys as algorithm names and values as the list of rewards obtained during training
    - a: The factor to aggregate the rewards, every b/a episodes a point will be plotted
    """
    plt.figure(figsize=(12, 6))

    for alg, rewards in algorithm_rewards.items():
        avg_rewards = [sum(rewards[i:i + a]) / a for i in range(0, len(rewards), a)]
        plt.plot(avg_rewards, '-o', label=alg)

    plt.title("Comparison of Algorithms")
    plt.xlabel(f"Every {a} Episodes")
    plt.ylabel("Average Reward")
    plt.legend(loc="upper right")  # Display the legend in the upper right corner
    plt.grid(True)
    plt.show()


def find_optimal_allocation(env):
    num_agents = env.num_agents
    all_permutations = permutations(range(num_agents), num_agents)

    best_reward = -np.inf
    best_allocation = None
    for perm in all_permutations:
        reward = sum([env.task_rewards[i][perm[i]] for i in range(num_agents)]) - sum(
            [env.agent_costs[i] for i in perm])
        if reward > best_reward:
            best_reward = reward
            best_allocation = perm

    return best_reward,best_allocation


# 5. Main Execution
if __name__ == "__main__":
    # ... setup configurations and environments
    # N_task = 6
    # N_agent = 6
    cfg = DQNConfig(6,6,True)
    env = MultiTaskEnvironment(cfg)
    agent_type = "DQN"
    # # for agent_type in ["DQN", "PPO", "TD3", "DDPG"]:
    # avg_reward = train(agent_type, cfg, env)
    # # result = test(agent_type, cfg, env)
    # best_reward,best_result = find_optimal_allocation(env)
    # print("best_reward:",best_reward)
    # print("best_result",best_result)
    cfg1 = DQNConfig(6,6,False)
    env1 = MultiTaskEnvironment(cfg1)
    average_reward, optical_allocate = test(agent_type,cfg1, env1)

    best_reward,best_result = find_optimal_allocation(env1)
    print("best_reward:",best_reward)
    print("best_result",best_result)

    # 寻找最佳超参数
    # study = optuna.create_study()
    # cfg = TD3Config(6,6,True)
    # env = MultiTaskEnvironment(cfg)
    # study.optimize(lambda trial: objective(trial, "TD3", cfg, env), n_trials=20)
    # print(study.best_params)
