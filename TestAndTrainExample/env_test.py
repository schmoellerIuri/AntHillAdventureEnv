import torch
from env_wrapper import AntEnvironmentWrapper
from DQN import DQN
import time
import matplotlib.pyplot as plt
import pandas as pd

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Testing will run on CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_size = 4

env = AntEnvironmentWrapper(False, env_size)
input_dim = env.observation_space.shape[0]  
output_dim = env.action_space.n 

network = DQN(input_dim, output_dim, env_size).to(device)
network.load_state_dict(torch.load(f'Models/{env_size}x{env_size}_model.pth'))
network.eval()

total_rewards = []

for episode in range(1000):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 100: 
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
                action = network(state_tensor.to(device)).argmax().item()
        next_state, reward, done = env.step(action)
        
        state = next_state
        total_reward += reward
        steps += 1

        if reward == -1: break

        #time.sleep(.1)
        
    print(f"Episode {episode + 1}/{100}, Total Reward: {total_reward}, Steps: {steps}")

    total_rewards.append(total_reward)

def moving_average(data, window_size):
    return pd.Series(data).rolling(window=window_size).mean()

# Define window size for smoothing
window_size = 50  # Adjust based on your preference

# Compute the moving average
smoothed_rewards = moving_average(total_rewards, window_size)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(total_rewards, label="Recompensas por episódio", alpha=0.4)
plt.plot(smoothed_rewards, label=f"Recompensas suavizadas (Tamanho da janela: {window_size})", color='red', linewidth=2)
plt.title("Progresso da recompensa com o tempo")
plt.xlabel("Episódios")
plt.ylabel("Recompensa total")
plt.legend()
plt.grid(True)
plt.show()

env.close()