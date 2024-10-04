import keyboard
import time
from env_wrapper import AntEnvironmentWrapper

env = AntEnvironmentWrapper(True, 8)

map_keys = {'w':0, 's':1, 'd':2, 'a':3, 'e':4, 'r':-1}

action = 0
acum_reward = 0
while 1:
    key = keyboard.read_key()
    if key == 'r':
        break
    if key in map_keys:
        action = map_keys[key]
        state, reward, done = env.step(action)
        acum_reward += reward
        time.sleep(0.15)
        print(reward)
        if done:
            print('Game Over')
            print('Acumulated Reward:', acum_reward)
            acum_reward = 0
            env.reset()
            
    
env.close()