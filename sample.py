import numpy as np
import torch
import optuna
import  os
import matplotlib.pyplot as plt
from sample_DQN import DQNAgent, train_dqn, test_dqn
from sample_PPO import PPOAgent, train_ppo, test_ppo
from sample_DDPG import DDPGAgent, train_ddpg, test_ddpg
from sample_TD3 import TD3Agent, train_td3,test_td3,ReplayBuffer
from itertools import permutations

class Config:
    def __init__(self,N_task,N_agent):
        # Environment parameters
        self.num_agents = N_agent
        self.num_tasks = N_task
        self.mode = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # DQN Agent parameters
        self.input_dim = self.num_agents  # This is set based on num_agents
        self.output_dim = self.num_tasks  # This is set based on num_tasks
        self.epsilon = 0.93
        self.gamma = 0.83
        self.batch_size = 32
        self.memory_size = 3500
        self.is_training = True

        self.max_episodes = 100000
        self.patience = 100000

        # Training parameters
        self.episodes = 100000
        self.num_iterations = 1
        self.max_steps = 3
        self.eps_decay = 0.955

        # Model saving/loading
        self.model_path = "dqn_model.pth"

        self.test_episodes = 50

def plot_results(episode_rewards, agent_type, title=None):
    """
    episode_rewards: list of average rewards per episode
    agent_type: string representing the type of agent (e.g., 'DQN')
    title: optional string for the plot title
    """
    plt.figure(figsize=(12,6))
    plt.plot(episode_rewards, label=f"{agent_type} Rewards per Episode")
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend(loc='upper left')
    if title:
        plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.show()


def objective(trial):
    # 1. Define the hyperparameters' range
    epsilon = trial.suggest_float("epsilon", 0.9, 1,step=0.01)
    gamma = trial.suggest_float("gamma", 0.8, 1, step=0.01)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    memory_size = trial.suggest_int("memory_size", 1000, 5000, step=500)
    eps_decay = trial.suggest_float("eps_decay", 0.9, 1, step=0.005)

    # 2. Use the hyperparameters in the Config
    cfg = Config(10,10)
    cfg.epsilon = epsilon
    cfg.gamma = gamma
    cfg.batch_size = batch_size
    cfg.memory_size = memory_size
    cfg.eps_decay = eps_decay
    print(f"Trial {trial.number} params: {trial.params}")

    # 3. Run the training process using these hyperparameters
    env = MultiTaskEnvironment(cfg)
    agent = DQNAgent(cfg)
    average_reward = train_dqn(cfg, agent, env)

    # 4. The objective is to maximize the average reward, so we return it as a negative value (since Optuna minimizes by default)
    return -average_reward
class MultiTaskEnvironment:
    def __init__(self, cfg):
        self.num_agents = cfg.num_agents
        self.num_tasks = cfg.num_tasks
        self.state = np.zeros(cfg.num_agents, dtype=np.int64)
        self.task_rewards,self.agent_costs = self._generate_probs(cfg.mode)
        self.assigned_tasks = []
        self.assigned_agents = []
        self.is_training = cfg.is_training
        self.total_profit = 0
        self.index = 1

    def _generate_probs(self, mode):
        if mode == 0:
            np.random.seed(0)
        elif mode != 1:
            raise ValueError("Invalid mode. Choose either 0 or 1.")

        rewards_agent = np.random.randint(10, 100, size=(self.num_agents, self.num_tasks))
        costs = np.random.randint(10, 50, size=self.num_agents)
        np.random.seed(None)
        return rewards_agent,costs

    def reset(self):
        self.state = np.zeros(self.num_agents, dtype=np.int64)
        self.assigned_tasks = []
        self.assigned_agents = []
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

            reward = self.index * max_reward_diff + (1-self.index) * self.total_profit / len(self.assigned_agents)

        # 为结束条件进行检查
        done = len(self.assigned_tasks) == self.num_tasks or agent_idx == self.num_agents
        # print("len(self.assigned_tasks)",len(self.assigned_tasks))
        # print("self.num_tasks",self.num_tasks)
        # print("len(self.assigned_agents)",len(self.assigned_agents))
        # print("self.num_agents",self.num_agents)
        return self.state, reward, done, {}


def run_training(agent_type, cfg, env):
    all_avg_rewards = []

    for iteration in range(cfg.num_iterations):
        if agent_type == "DQN":
            print(" DQN iteration:",iteration+1)
            agent = DQNAgent(cfg)
            avg_reward = train_dqn(cfg, agent, env)
        elif agent_type == "PPO":
            print(" PPO iteration:",iteration)
            agent = PPOAgent(cfg)
            avg_reward = train_ppo(cfg, agent, env)
        elif agent_type == "DDPG":
            print(" DDPG iteration:",iteration)
            agent = DDPGAgent(cfg)
            avg_reward = train_ddpg(cfg, agent, env)
        elif agent_type == "TD3":
            print(" TD3 iteration:", iteration)
            agent = TD3Agent(cfg)
            replay_buffer = ReplayBuffer(1e6)
            avg_reward = train_td3(cfg, agent, env,replay_buffer)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        all_avg_rewards.append(avg_reward)  # store the average reward from this iteration

    return all_avg_rewards


def plot_combined_results(results, title="Training Curve"):
    plt.figure(figsize=(12, 6))

    for agent_type, rewards in results.items():
        plt.plot(rewards, label=f"{agent_type} Rewards per Episode")

    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig("comparison_plot.png")
    plt.show()


def find_optimal_allocation(env):
    num_agents = env.num_agents
    all_permutations = permutations(range(num_agents), num_agents)

    best_reward = -np.inf
    best_allocation = None
    for perm in all_permutations:
        reward = sum([env.task_rewards[i][perm[i]] for i in range(num_agents)]) - sum([env.agent_costs[i] for i in perm])
        if reward > best_reward:
            best_reward = reward
            best_allocation = perm

    return best_allocation, best_reward

def evaluate_and_print(agent_type, average_reward):
    print(f"{agent_type} Average reward:", average_reward)


if __name__ == "__main__":

    print(os.getcwd())

    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # print(torch.version.cuda)
    cfg = Config(6,6)
    env = MultiTaskEnvironment(cfg)
    agents = ["DQN", "PPO", "TD3", "DDPG"]
    all_results = {}
    for agent_type in agents:
        rewards = run_training(agent_type, cfg, env)
        all_results[agent_type] = rewards
    plot_combined_results(all_results, "Comparison of Different Algorithms")
    best_allocation, best_reward = find_optimal_allocation(env)
    print("True Optimal Allocation:", best_allocation)
    print("True Optimal Reward:", best_reward)
    # rewards = run_training("DQN", cfg, env)

    # Training DQN
    # agent_DQN = DQNAgent(cfg)
    # average_reward_DQN = train_dqn(cfg, agent_DQN, env)
    # evaluate_and_print("DQN", average_reward_DQN)
    # # Training DDPG
    # agent_DDPG = DDPGAgent(cfg)
    # average_reward_DDPG = train_ddpg(cfg, agent_DDPG, env)
    # evaluate_and_print("DDPG", average_reward_DDPG)


    # replay_buffer = ReplayBuffer(1e6)
    #
    # agent_TD3 =TD3Agent(cfg)
    # average_reward_TD3 = train_td3(cfg, agent_TD3,env,replay_buffer)


    # agents = ["DQN", "PPO", "DDPG","TD3"]
    #
    # # Collect results for each algorithm
    # all_results = {}
    # for agent_type in agents:
    #     rewards = run_training(agent_type, cfg, env)
    #     all_results[agent_type] = rewards
    #
    # plot_combined_results(all_results, "Comparison of Different Algorithms")
    #
    # best_allocation, best_reward = find_optimal_allocation(env)
    # print("True Optimal Allocation:", best_allocation)
    # print("True Optimal Reward:", best_reward)
    #   训练超参数
    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=50)  # You can change the number of trials based on your preference
    #
    # print("Best hyperparameters:", study.best_params)
    # print("Best average reward:", -study.best_value)

    # cfg = Config()
    # env = MultiTaskEnvironment(cfg)
    #
    # # Training DQN
    # agent_DQN = DQNAgent(cfg)
    # average_reward_DQN = train_dqn(cfg, agent_DQN, env)
    # evaluate_and_print("DQN", average_reward_DQN)
    # #
    # #  Training PPO
    # agent_PPO = PPOAgent(cfg)
    # average_reward_PPO = train_ppo(cfg, agent_PPO, env)
    # evaluate_and_print("PPO", average_reward_PPO)
    #

    #
    # best_allocation, best_reward = find_optimal_allocation(env)
    # print("True Optimal Allocation:", best_allocation)
    # print("True Optimal Reward:", best_reward)
    #   保存模型
    # torch.save(agent.q_net.state_dict(), "dqn_model.pth")
    # agent.q_net.load_state_dict(torch.load("dqn_model.pth"))
    # agent.q_net.eval()
    #
    # cfg1 = Config()
    # cfg1.is_training = False
    #
    # test_env = MultiTaskEnvironment(cfg1)
    #
    # optimal_actions,optimal_rewards = test_dqn(agent, test_env)
    #
    # best_allocation, best_reward = find_optimal_allocation(env)
    #
    #
    # print("True Optimal Allocation:", best_allocation)
    # print("True Optimal Reward:", best_reward)
    # print("Optimal allocation by DQN:", optimal_actions)
    # print("Optimal reward by DQN:", optimal_rewards)