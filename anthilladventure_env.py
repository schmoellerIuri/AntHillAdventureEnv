import time
import numpy as np
import ursina as us
import keyboard

class AntEnvironment():
    def __init__(self, render_gui=False) :
        self.render_gui = render_gui

        self.ant_position = (np.random.randint(0,7),np.random.randint(0,7))

        self.previous_positions = set()

        leaves = self.fill_positions(3)
        trashes = self.fill_positions(2)
        sticks = self.fill_positions(1)
        rocks = self.fill_positions(1)
        anthill_position = self.fill_positions(1)

        self.carried_object = np.array([1,0,0,0,0])

        self.objects_positions = [leaves,trashes,sticks,rocks,anthill_position]

        self.state = {
            'ant_position' : self.ant_position,
            'objects_positions' : self.objects_positions,
            'carried_object' : self.carried_object
        }

        self.first_state = self.state

        self.buildWindow()

    def close(self):
        if self.render_gui:
            self.app.destroy()
    
    def fill_positions(self, quantity):
        positions = set()
        for _ in range(quantity):
            pos = (np.random.randint(0,7),np.random.randint(0,7))
            while (pos[0] == self.ant_position[0] and pos[1] == self.ant_position[1]) or pos in self.previous_positions:
                pos = (np.random.randint(0,7),np.random.randint(0,7))
            
            positions.add(pos)
            self.previous_positions.add(pos)

        return positions
    
    def step(self, action):
        if self.state['carried_object'][0] == 1:
            reward = -1
        elif self.state['carried_object'][1] == 1 or self.state['carried_object'][3] == 1:
            reward = -.5
        else:
            reward = -2

        picked = False
        dropped = False
        item_dropped = -1

        pos = self.state['ant_position']
        x = pos[0]
        y = pos[1]

        if action == 0:
            if y > 0:
                self.state['ant_position'] = (x,y-1)
        elif action == 1:
            if y < 7:
                self.state['ant_position'] = (x,y+1)
        elif action == 2:
            if x < 7:
                self.state['ant_position'] = (x+1,y)
        elif action == 3:
            if x > 0:
                self.state['ant_position'] = (x-1,y)
        elif action == 4:
            if self.state['carried_object'][0] == 1:
                if self.ant_can_pick_object(0): #picked a leaf   
                    picked = True                 
                    self.change_carried_object(1)
                    reward = 10
                elif self.ant_can_pick_object(1): #picked a trash                    
                    picked = True
                    self.change_carried_object(2)
                    reward = -10
                elif self.ant_can_pick_object(2): #picked a stick                    
                    picked = True
                    self.change_carried_object(3)
                    reward = 10
                elif self.ant_can_pick_object(3): #picked a rock                    
                    picked = True
                    self.change_carried_object(4)
                    reward = -5
                else:   #tried to pick nothing                    
                    reward = -2
            else:
                if self.ant_can_drop_object():
                    item_dropped = np.argmax(self.state['carried_object'])
                    if item_dropped == 1 or item_dropped == 3:
                        reward = 20 if self.state['ant_position'] in self.state['objects_positions'][-1] else -5
                    else:
                        reward = -20 if self.state['ant_position'] in self.state['objects_positions'][-1] else -.5 
                    dropped = True
                    self.drop_object()
                else:
                    reward = -2
            
        done = len(self.state['objects_positions'][0]) == 0 and len(self.state['objects_positions'][2]) == 0 and self.state['carried_object'][0] == 1

        self.update(picked,dropped,item_dropped,action)

        self.app.step()
        self.app.step()

        return self.state, reward, done
                    
    def ant_can_pick_object(self, obj_index):
        return self.state['ant_position'] in self.state['objects_positions'][obj_index]

    def change_carried_object(self, obj_index):
        self.state['objects_positions'][obj_index-1].remove(self.state['ant_position'])
        self.state['carried_object'][0] = 0
        self.state['carried_object'][obj_index] = 1
    
    def ant_can_drop_object(self):
        for i in range(len(self.state['objects_positions']) - 1): #check every object but the anthill, cause you can actually drop an item in the anthill
            if self.state['ant_position'] in self.state['objects_positions'][i]:
                return False
        return True

    def drop_object(self):
        index = np.argmax(self.state['carried_object']) # index of the current object that is being carried 
        if self.state['ant_position'] not in self.state['objects_positions'][-1]:
            self.state['objects_positions'][index-1].add(self.state['ant_position'])
        self.state['carried_object'][index] = 0
        self.state['carried_object'][0] = 1
    
    def buildWindow(self):
        self.app = us.Ursina()

        us.window.size = (800,800)
        us.window.color = us.color.rgb(102/255, 51/255, 0)
        us.camera.orthographic = True
        us.camera.fov = 1

        self.object_renders = {}

        for i in range(8):
            for j in range(8):
                texture = 'images/ground.jpg'

                if (i,j) in self.state['objects_positions'][-1]:
                    texture = 'images/anthill.jpg'

                us.Entity(scale=(1/8, 1/8), model='quad', texture=texture, origin_x=0, origin_y=0, x=-.438 + 1/8*i, y=.438 - 1/8*j, z = 0)

                if (i,j) == self.ant_position:
                    self.ant_render = us.Entity(scale=(1/10, 1/10), model='quad', texture='images/ant.png', origin_x=0, origin_y=0, x=-.438 + 1/8*i, y=.438 - 1/8*j, z = -1)

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
                    self.object_renders[(i,j)] = us.Entity(scale=(1/15, 1/15), model='quad', texture=f'images/{file_name}.png', origin_x=0, origin_y=0, x=-.438 + 1/8*i, y=.438 - 1/8*j, z = -1, object_type = file_name)    

        self.rendered = True
        for _ in range(10):
            self.app.step()
    
    def update(self, picked, dropped, itemDropped,lastAction = -1):
            self.ant_render.x = -.438 + 1/8*self.state['ant_position'][0]
            self.ant_render.y = .438 - 1/8*self.state['ant_position'][1]

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
                    self.object_renders[(i,j)] = us.Entity(scale=(1/15, 1/15), model='quad', texture=f'images/{file_name}.png', origin_x=0, origin_y=0, x=-.438 + 1/8*i, y=.438 - 1/8*j, z = -1, object_type = file_name)

    def reset(self):
        #self.state = self.first_state
        #self.app.destroy()
        #self.buildWindow()
        pass
    
    def close(self):
        self.app.destroy()
        

ant = AntEnvironment(render_gui=True)

map_keys = {'w':0, 's':1, 'd':2, 'a':3, 'e':4, 'r':-1}

action = 0
acum_reward = 0
while 1:
    key = keyboard.read_key()
    if key == 'r':
        break
    if key in map_keys:
        action = map_keys[key]
        state, reward, done = ant.step(action)
        acum_reward += reward
        time.sleep(0.15)
        print(reward)
        if done:
            print('Game Over')
            print('Acumulated Reward:', acum_reward)
            acum_reward = 0
            #ant.reset()
            break
    
ant.close()