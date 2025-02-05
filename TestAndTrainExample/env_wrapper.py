from ant_env import AntEnvironment
import numpy as np

class AntEnvironmentWrapper:
    def __init__(self, render=False, n=4):

        self.env = AntEnvironment(render,n)
        self.n = n
        self.action_space = type('ActionSpace', (), {'n': 5})()  # 5 ações: 0-4
        
        shape = (6 * n**2) + 5

        self.observation_space = type('ObservationSpace', (), {'shape': (shape,)})()

    def reset(self):
        state = self.env.reset()
        return self.preprocess_state(state)
    
    def step(self, action):
        state, reward, done = self.env.step(action)
        return self.preprocess_state(state), reward, done
    
    def close(self):
        self.env.close()

    def preprocess_state(self, state):
        ant_position_encoded = np.zeros(self.n**2)

        x,y = state.ant_position
        ant_position_encoded[x*(self.n) + y] = 1
        
        objects_positions_encoded = np.zeros(self.n**2 * 5)
        i = 0

        for positions in state.objects_positions:
            for position in positions:
                obj_offset = self.n**2
                pos_offset = self.n
                objects_positions_encoded[i*obj_offset+position[0]*pos_offset+position[1]] = i + 2
            i+=1

        state_encoded = np.concatenate([ant_position_encoded,objects_positions_encoded,state.carried_object])

        return state_encoded