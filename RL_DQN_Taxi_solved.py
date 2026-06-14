"""
DQN solution for the Gymnasium Taxi environment.

This is intentionally more commented than normal production code because the
goal is to make the deep-RL pieces easy to study:

- one-hot state vectors
- epsilon-greedy exploration
- replay memory
- policy network
- target network
- Bellman target update

Note: Taxi has only 500 discrete states, so tabular Q-learning is the simpler
and more natural solution. DQN is useful here as a learning exercise for how we
would handle larger state spaces where a Q-table is no longer practical.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import deque, namedtuple
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# A single saved transition from the agent's experience.
#
# state:      the old Taxi state, an integer from 0 to 499
# action:     the action taken in that state, an integer from 0 to 5
# reward:     the immediate reward returned by env.step(action)
# next_state: the state reached after taking the action
# done:       True if the episode ended, either naturally or by time limit
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


@dataclass
class DQNConfig:
    """All important training choices in one place."""

    seed: int = 7
    episodes: int = 500
    max_steps_per_episode: int = 200

    # Q-learning discount factor.
    # A value close to 1 means "future rewards matter almost as much as now".
    gamma: float = 0.99

    # Optimizer settings for the neural network.
    learning_rate: float = 1e-3
    batch_size: int = 64

    # Replay buffer settings.
    # The buffer stores old transitions so we can train on a random mix instead
    # of only the latest step. That breaks correlations between consecutive
    # states and makes DQN training much more stable.
    replay_capacity: int = 50_000
    min_replay_size: int = 500

    # For this small Taxi problem we train after every environment step once
    # the replay buffer has enough examples. That gives the network many chances
    # to correct its Q-values while the assignment still runs in a reasonable
    # amount of time.
    train_frequency: int = 1

    # Target-network update frequency, measured in environment steps.
    # The policy network changes every optimizer step. The target network is a
    # slower copy, so the Bellman targets are less jumpy.
    target_update_steps: int = 250

    # Epsilon-greedy exploration schedule.
    # Early in training epsilon is high, so the agent tries lots of random
    # actions. Later epsilon approaches min_epsilon, so the agent mostly trusts
    # the Q-values learned by the network.
    max_epsilon: float = 1.0
    min_epsilon: float = 0.05
    decay_rate: float = 0.0015

    # Evaluation uses greedy actions only, so epsilon = 0.
    eval_episodes: int = 100


class ReplayMemory:
    """Fixed-size memory of past transitions."""

    def __init__(self, capacity: int) -> None:
        self.memory: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class QNetwork(nn.Module):
    """
    Small neural network that estimates Q(s, a).

    Input:
        one-hot encoded state vector of length 500

    Output:
        6 Q-values, one per Taxi action:
        0=south, 1=north, 2=east, 3=west, 4=pickup, 5=dropoff
    """

    def __init__(self, n_states: int, n_actions: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def make_taxi_env(render_mode: str | None = None) -> gym.Env:
    """
    Create Taxi in a way that works across Gymnasium versions.

    Newer Gymnasium versions renamed Taxi-v3 to Taxi-v4. The task is the same
    for our purposes, so we simply try v4 first and then fall back to v3.
    """

    for env_name in ("Taxi-v4", "Taxi-v3"):
        try:
            return gym.make(env_name, render_mode=render_mode)
        except Exception:
            pass
    raise RuntimeError("Could not create Taxi-v4 or Taxi-v3. Try installing gymnasium[toy-text].")


def set_seed(seed: int, env: gym.Env) -> None:
    """Seed Python, NumPy, PyTorch, and the environment action space."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.action_space.seed(seed)


def states_to_one_hot(states: torch.Tensor, n_states: int) -> torch.Tensor:
    """
    Convert Taxi state integers into one-hot vectors.

    A neural network should not receive the raw integer state directly. The
    integer 499 is not "larger" than state 12 in a meaningful geometric sense;
    it is just an ID. One-hot encoding treats every state ID as its own category.
    """

    return F.one_hot(states.long(), num_classes=n_states).float()


def epsilon_for_episode(episode: int, config: DQNConfig) -> float:
    """Exponential decay from max_epsilon down toward min_epsilon."""

    return config.min_epsilon + (config.max_epsilon - config.min_epsilon) * math.exp(
        -config.decay_rate * episode
    )


def select_action(
    state: int,
    epsilon: float,
    env: gym.Env,
    policy_net: QNetwork,
    n_states: int,
    device: torch.device,
) -> int:
    """
    Epsilon-greedy action selection.

    With probability epsilon, explore with a random action.
    Otherwise, exploit by choosing the action with the largest predicted Q-value.
    """

    if random.random() < epsilon:
        return int(env.action_space.sample())

    # We are only choosing an action, not training, so gradients are unnecessary.
    with torch.no_grad():
        state_tensor = torch.tensor([state], device=device)
        state_one_hot = states_to_one_hot(state_tensor, n_states)
        q_values = policy_net(state_one_hot)
        return int(q_values.argmax(dim=1).item())


def train_step(
    memory: ReplayMemory,
    policy_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    config: DQNConfig,
    n_states: int,
    device: torch.device,
) -> float | None:
    """
    Run one DQN optimizer step from a random replay-memory batch.

    This is the heart of DQN. We train the policy network to make its current
    Q-value estimate closer to the Bellman target:

        target = reward + gamma * max_a Q_target(next_state, a)

    If the episode is done, there is no future reward, so:

        target = reward
    """

    if len(memory) < config.min_replay_size:
        return None

    transitions = memory.sample(config.batch_size)
    batch = Transition(*zip(*transitions))

    states = torch.tensor(batch.state, device=device)
    actions = torch.tensor(batch.action, device=device)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
    next_states = torch.tensor(batch.next_state, device=device)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=device)

    state_vectors = states_to_one_hot(states, n_states)
    next_state_vectors = states_to_one_hot(next_states, n_states)

    # policy_net(state) returns Q-values for every action.
    # gather(...) selects the Q-value for the action we actually took.
    predicted_q = policy_net(state_vectors).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # The target network supplies the future-value estimate.
        # Because it changes slowly, the target is more stable than using the
        # policy network on both sides of the equation.
        best_next_q = target_net(next_state_vectors).max(dim=1).values

        # If done == 1, the multiplier becomes zero and the future term vanishes.
        target_q = rewards + config.gamma * (1.0 - dones) * best_next_q

    loss = F.smooth_l1_loss(predicted_q, target_q)

    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping is a small safety guard against rare large updates.
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)

    optimizer.step()
    return float(loss.item())


def train_dqn(config: DQNConfig) -> tuple[QNetwork, list[float], list[float], gym.Env]:
    """Train the DQN agent and return the trained policy network."""

    env = make_taxi_env()
    set_seed(config.seed, env)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Taxi is tiny, so CPU is usually faster and simpler than moving tensors to
    # a GPU/MPS device. Keeping the device explicit still makes the code clear.
    device = torch.device("cpu")

    policy_net = QNetwork(n_states, n_actions).to(device)
    target_net = QNetwork(n_states, n_actions).to(device)

    # At the start, both networks must agree exactly.
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=config.learning_rate)
    memory = ReplayMemory(config.replay_capacity)

    episode_rewards: list[float] = []
    losses: list[float] = []
    global_step = 0

    state, _ = env.reset(seed=config.seed)

    for episode in range(1, config.episodes + 1):
        state, _ = env.reset()
        epsilon = epsilon_for_episode(episode, config)
        total_reward = 0.0

        for _ in range(config.max_steps_per_episode):
            global_step += 1

            action = select_action(state, epsilon, env, policy_net, n_states, device)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # In Gymnasium, terminated means the task ended naturally.
            # truncated means a wrapper stopped the episode, usually due to a
            # step limit. For training, either one means "no more steps now".
            done = terminated or truncated

            memory.push(Transition(state, action, reward, next_state, done))

            loss = None
            if global_step % config.train_frequency == 0:
                loss = train_step(memory, policy_net, target_net, optimizer, config, n_states, device)
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

            # Every so often, copy the learned policy weights into the target
            # network. This is the "two networks" trick that stabilizes DQN.
            if global_step % config.target_update_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        episode_rewards.append(total_reward)

        if episode % 100 == 0:
            avg_reward = float(np.mean(episode_rewards[-100:]))
            print(
                f"Episode {episode:4d} | "
                f"avg reward last 100: {avg_reward:6.2f} | "
                f"epsilon: {epsilon:5.3f} | "
                f"replay size: {len(memory):5d}"
            )

    env.close()
    return policy_net, episode_rewards, losses, make_taxi_env()


def evaluate_agent(
    policy_net: QNetwork,
    env: gym.Env,
    episodes: int,
    max_steps: int,
    seed: int,
) -> tuple[float, float, int]:
    """Evaluate greedily, with no random exploration."""

    n_states = env.observation_space.n
    device = torch.device("cpu")
    rewards: list[float] = []
    steps_taken: list[int] = []
    successes = 0

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0

        for step in range(max_steps):
            action = select_action(
                state=state,
                epsilon=0.0,
                env=env,
                policy_net=policy_net,
                n_states=n_states,
                device=device,
            )
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated:
                successes += 1
                break
            if truncated:
                break

        rewards.append(total_reward)
        steps_taken.append(step + 1)

    return float(np.mean(rewards)), float(np.mean(steps_taken)), successes


def plot_training(episode_rewards: list[float], losses: list[float]) -> None:
    """Plot learning progress."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(episode_rewards, alpha=0.35, label="episode reward")
    if len(episode_rewards) >= 100:
        moving_avg = np.convolve(episode_rewards, np.ones(100) / 100, mode="valid")
        axes[0].plot(range(99, len(episode_rewards)), moving_avg, label="100-episode average")
    axes[0].set_title("DQN Taxi training reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    axes[1].plot(losses, alpha=0.6)
    axes[1].set_title("DQN loss")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Smooth L1 loss")

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PyTorch DQN agent on Taxi.")
    parser.add_argument("--episodes", type=int, default=DQNConfig.episodes)
    parser.add_argument("--eval-episodes", type=int, default=DQNConfig.eval_episodes)
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plots.")
    args = parser.parse_args()

    config = DQNConfig(episodes=args.episodes, eval_episodes=args.eval_episodes)
    policy_net, episode_rewards, losses, env = train_dqn(config)

    avg_reward, avg_steps, successes = evaluate_agent(
        policy_net=policy_net,
        env=env,
        episodes=config.eval_episodes,
        max_steps=config.max_steps_per_episode,
        seed=config.seed + 10_000,
    )
    env.close()

    print("\nEvaluation with epsilon = 0, so the agent is greedy:")
    print(f"  average reward : {avg_reward:.2f}")
    print(f"  average steps  : {avg_steps:.1f}")
    print(f"  success rate   : {successes}/{config.eval_episodes}")

    if not args.no_plot:
        plot_training(episode_rewards, losses)


if __name__ == "__main__":
    main()
