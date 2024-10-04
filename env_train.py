import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from env_wrapper import AntEnvironmentWrapper
from DQN import DQN

# Hyperparameters
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.9999
lr = 0.00005
batch_size = 128
target_update = 10
memory_size = 10000
episodes = 30000

def select_action(state, policy_net, epsilon, action_space_n, device):
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state.to(device)).argmax().item()
    else:
        return random.randrange(action_space_n)

def optimize_model(policy_net, target_net, memory, optimizer, criterion, gamma, batch_size, device):
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(np.array(actions), dtype=torch.long).unsqueeze(1).to(device)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(device)
    
    # Clip rewards
    rewards = torch.clamp(rewards, -1, 1)
    
    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
    expected_q_values = rewards + (gamma * next_q_values * (1 - dones))
    
    loss = criterion(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1)
    optimizer.step()

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = AntEnvironmentWrapper(False,8)
input_dim = env.observation_space.shape[0]  
output_dim = env.action_space.n  

print(input_dim)
print(output_dim)

policy_net = DQN(input_dim, output_dim).to(device)

target_net = DQN(input_dim, output_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
criterion = nn.SmoothL1Loss()  # Huber loss for stability

memory = deque(maxlen=memory_size)
epsilon = epsilon_start

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    sync_steps = 0
    
    while not done and steps < 1000:  # Add step limit to prevent infinite loops
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = select_action(state_tensor, policy_net, epsilon, env.action_space.n, device)
        next_state, reward, done = env.step(action)
        memory.append((state, action, reward, next_state, done))
        
        state = next_state
        total_reward += reward
        steps += 1
        sync_steps += 1
        
        if sync_steps == 10:
            optimize_model(policy_net, target_net, memory, optimizer, criterion, gamma, batch_size, device)
            sync_steps = 0
    
    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Steps: {steps}, Epsilon: {epsilon:.4f}")
    # Save the model every 100 episodes
    if (episode + 1) % 100 == 0:
        torch.save(policy_net.state_dict(), f'dqn_model_episode_{episode+1}.pth')

env.close()