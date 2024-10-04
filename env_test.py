import torch
from env_wrapper import AntEnvironmentWrapper
from DQN import DQN
import time

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Testing will run on CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = AntEnvironmentWrapper(True,8)
input_dim = env.observation_space.shape[0]  # 101
output_dim = env.action_space.n  # 5

network = DQN(input_dim, output_dim).to(device)
network.load_state_dict(torch.load('Models/8x8_model.pth',weights_only=True))
network.eval()

for episode in range(100):
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

        time.sleep(.1)
        
    print(f"Episode {episode + 1}/{100}, Total Reward: {total_reward}, Steps: {steps}")

env.close()