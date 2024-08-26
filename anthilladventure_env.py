#import time
import numpy as np
import ursina as us
#import keyboard

class AntEnvironment():
    def __init__(self) :
        self.state = self.__get_random_state()

        self.rendered = False

        self.__buildWindow()
    
    def __get_random_state(self):
        self.ant_position = (np.random.randint(0,3),np.random.randint(0,3))

        previous_positions = set()

        leaves, previous_positions = self.__get_random_positions(3, previous_positions)
        trashes, previous_positions = self.__get_random_positions(2, previous_positions)
        sticks, previous_positions = self.__get_random_positions(1, previous_positions)
        rocks, previous_positions = self.__get_random_positions(1, previous_positions)
        anthill, _ = self.__get_random_positions(1, previous_positions)

        carried_object = np.array([1,0,0,0,0])

        objects_positions = [leaves,trashes,sticks,rocks,anthill]

        return {
            'ant_position' : self.ant_position,
            'objects_positions' : objects_positions,
            'carried_object' : carried_object
        }

    def __get_random_positions(self, quantity, previous_positions):
        positions = set()
        for _ in range(quantity):
            pos = (np.random.randint(0,3),np.random.randint(0,3))
            while (pos[0] == self.ant_position[0] and pos[1] == self.ant_position[1]) or pos in previous_positions:
                pos = (np.random.randint(0,3),np.random.randint(0,3))
            
            positions.add(pos)
            previous_positions.add(pos)

        return positions, previous_positions

    def __buildWindow(self):

        if not self.rendered:
            self.app = us.Ursina()

            us.window.size = (400,400)
            us.window.color = us.color.rgb(102/255, 51/255, 0)
            us.camera.orthographic = True
            us.camera.fov = 1

        self.object_renders = {}
        self.ground_renders = []

        for i in range(4):
            for j in range(4):
                texture = 'images/ground.jpg'

                if (i,j) in self.state['objects_positions'][-1]:
                    texture = 'images/anthill.jpg'

                self.ground_renders.append(us.Entity(scale=(1/4, 1/4), model='quad', texture=texture, origin_x=0, origin_y=0, x=-.375 + 1/4*i, y=.375 - 1/4*j, z = 0))

                if (i,j) == self.ant_position:
                    self.ant_render = us.Entity(scale=(1/5, 1/5), model='quad', texture='images/ant.png', origin_x=0, origin_y=0, x=-.375 + 1/4*i, y=.375 - 1/4*j, z = -1)

                file_name = ''
                if (i,j) in self.state['objects_positions'][0]:
                    file_name = 'leaf'
                if (i,j) in self.state['objects_positions'][1]:
                    file_name = 'trash'
                if (i,j) in self.state['objects_positions'][2]:
                    file_name = 'sticks'
                if (i,j) in self.state['objects_positions'][3]:
                    file_name = 'rock'
                    
                if len(file_name) > 0:
                    self.object_renders[(i,j)] = us.Entity(scale=(2/15, 2/15), model='quad', texture=f'images/{file_name}.png', origin_x=0, origin_y=0, x=-.375 + 1/4*i, y=.375 - 1/4*j, z = -1, object_type = file_name)    

        self.rendered = True
        for _ in range(10):
            self.app.step()
    
    def step(self, action):
        if action not in range(5):
            raise TypeError("Action should be an integer in range [0,4]")
        
        carried_obj = np.argmax(self.state['carried_object'])

        if carried_obj == 0:
            reward = -.8
        elif carried_obj == 1 or carried_obj == 3:
            reward = -.2
        else:
            reward = -1

        picked = False
        dropped = False
        item_dropped = -1

        pos_before = self.state['ant_position']
        
        if action in range(4):
            pos = self.__move_ant(action)
            reward = -1 if pos_before == pos else reward
        elif action == 4:
            if self.state['carried_object'][0] == 1:
                object_to_pick = self.__get_object_to_pick()

                if object_to_pick != -1: 
                    picked = True                 
                    self.__change_carried_object(object_to_pick)
                    reward = -.1 if object_to_pick == 1 or object_to_pick == 3 else -1
                else:   #tried to pick nothing                    
                    reward = -2
            else:
                if self.__ant_can_drop_object():
                    item_dropped = np.argmax(self.state['carried_object'])
                    if item_dropped == 1 or item_dropped == 3:
                        reward = 20 if self.state['ant_position'] in self.state['objects_positions'][-1] else -2
                    else:
                        reward = -20 if self.state['ant_position'] in self.state['objects_positions'][-1] else -.5 
                    dropped = True
                    self.__drop_object()
                else: #tried to do an illegal drop
                    reward = -2
            
        done = len(self.state['objects_positions'][0]) == 0 and len(self.state['objects_positions'][2]) == 0 and self.state['carried_object'][0] == 1

        self.__update(picked,dropped,item_dropped,action)

        self.app.step()
        self.app.step()

        return self.state, reward, done

    def __move_ant(self, action):
        x,y = self.state['ant_position']
        if action == 0 and y > 0:
            self.state['ant_position'] = (x,y-1)
        if action == 1 and y < 3:
            self.state['ant_position'] = (x,y+1)
        if action == 2 and x < 3:
            self.state['ant_position'] = (x+1,y)
        if action == 3 and x > 0:
            self.state['ant_position'] = (x-1,y)

        return self.state['ant_position']
                    
    def __get_object_to_pick(self):
        for i in range(4):
            if self.state['ant_position'] in self.state['objects_positions'][i]:
                return i + 1
        return -1

    def __change_carried_object(self, obj_index):
        self.state['objects_positions'][obj_index-1].remove(self.state['ant_position'])
        self.state['carried_object'][0] = 0
        self.state['carried_object'][obj_index] = 1
    
    def __ant_can_drop_object(self):
        for i in range(len(self.state['objects_positions']) - 1): #check every object but the anthill, cause you can actually drop an item in the anthill
            if self.state['ant_position'] in self.state['objects_positions'][i]:
                return False
        return True

    def __drop_object(self):
        index = np.argmax(self.state['carried_object']) # index of the current object that is being carried 
        if self.state['ant_position'] not in self.state['objects_positions'][-1]:
            self.state['objects_positions'][index-1].add(self.state['ant_position'])
        self.state['carried_object'][index] = 0
        self.state['carried_object'][0] = 1
    
    def __update(self, picked, dropped, itemDropped,lastAction = -1):
            self.ant_render.x = -.375 + 1/4*self.state['ant_position'][0]
            self.ant_render.y = .375 - 1/4*self.state['ant_position'][1]

            rotations = {0 : 0, 1 : 180, 2 : 90, 3 : -90}

            if lastAction in range(4):
                self.ant_render.rotation_z = rotations[lastAction]
            
            names = {0 : 'leaf', 1: 'trash', 2: 'sticks', 3: 'rock'}
            
            i = self.state['ant_position'][0]
            j = self.state['ant_position'][1]

            if picked:
                file_name = self.object_renders[(i,j)].object_type
                self.ant_render.texture = f'images/antwith{file_name}.png'
                self.object_renders[(i,j)].disable()
                _ = self.object_renders.pop((i,j))
            if dropped:
                self.ant_render.texture = 'images/ant.png'
                file_name = names[itemDropped-1]
                if (i,j) not in self.state['objects_positions'][-1]:
                    self.object_renders[(i,j)] = us.Entity(scale=(2/15, 2/15), model='quad', texture=f'images/{file_name}.png', origin_x=0, origin_y=0, x=-.375 + 1/4*i, y=.375 - 1/4*j, z = -1, object_type = file_name)

    def reset(self):
        for entity in self.object_renders.values():
            entity.disable()

        for item in self.ground_renders:
            item.disable()

        self.ground_renders.clear()

        self.object_renders.clear()

        self.ant_render.disable()

        self.state = self.__get_random_state()

        self.__buildWindow()

        return self.state

    def close(self):
        self.app.destroy()
        

env = AntEnvironment()

'''
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
'''