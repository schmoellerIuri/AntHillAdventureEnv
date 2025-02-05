import numpy as np
import ursina as us

class State():
    def __init__(self, ant_position, objects_positions, carried_object):
        self.ant_position = ant_position
        self.objects_positions = objects_positions
        self.carried_object = carried_object

class AntEnvironment():
    def __init__(self, render=False, grid_size=4) :
        if (grid_size < 4 or grid_size > 8): raise TypeError('grid_size must be an integer in range [4,9]')

        self.n = grid_size

        self.state = self.__get_random_state()

        self.render = render

        self.objects_names = {0 : 'leaf', 1: 'trash', 2: 'sticks', 3: 'rock'}

        if self.render:
            self.rendered = False
            self.__buildWindow()
    
    def __get_random_state(self):
        self.ant_position = (np.random.randint(0,self.n-1),np.random.randint(0,self.n-1))

        previous_positions = set()

        leaves, previous_positions = self.__get_random_positions(3, previous_positions)
        trashes, previous_positions = self.__get_random_positions(2, previous_positions)
        sticks, previous_positions = self.__get_random_positions(1, previous_positions)
        rocks, previous_positions = self.__get_random_positions(1, previous_positions)
        anthill, _ = self.__get_random_positions(1, previous_positions)

        carried_object = np.array([1,0,0,0,0])

        objects_positions = [leaves,trashes,sticks,rocks,anthill]

        return State(self.ant_position, objects_positions, carried_object)

    def __get_random_positions(self, quantity, previous_positions):
        positions = set()
        for _ in range(quantity):
            pos = (np.random.randint(0,self.n-1),np.random.randint(0,self.n-1))
            while (pos[0] == self.ant_position[0] and pos[1] == self.ant_position[1]) or pos in previous_positions:
                pos = (np.random.randint(0,self.n-1),np.random.randint(0,self.n-1))
            
            positions.add(pos)
            previous_positions.add(pos)

        return positions, previous_positions

    def __buildWindow(self):

        if not self.rendered:
            self.__set_window_configuration()
            self.object_renders = {}
            self.ground_renders = []

        self.__build_ground_renders()

        self.__build_objects_renders()  

        self.rendered = True
        for _ in range(10):
            self.app.step()
    
    def __set_window_configuration(self):
        self.app = us.Ursina()

        us.window.size = (self.n * 100,self.n * 100)
        us.window.color = us.color.rgb(102/255, 51/255, 0)
        us.camera.orthographic = True
        us.camera.fov = 1

        self.ant_offset = .5 - (1/(self.n*2))
    
    def __build_ground_renders(self):
        for i in range(self.n):
            for j in range(self.n):
                texture = 'images/ground.jpg'

                if (i,j) in self.state.objects_positions[-1]:
                    texture = 'images/anthill.jpg'

                if (i,j) == self.ant_position:
                    self.ant_render = us.Entity(scale=(1/self.n, 1/self.n), model='quad', texture='images/ant.png', origin_x=0, origin_y=0, x=-self.ant_offset + (1/self.n)*i, y=self.ant_offset - (1/self.n)*j, z = -1)

                self.ground_renders.append(us.Entity(scale=(1/self.n, 1/self.n), model='quad', texture=texture, origin_x=-.5, origin_y=.5, x=-.5 + (1/self.n)*i, y=.5 - (1/self.n)*j, z = 0))

    def __build_objects_renders(self):
        for i,positions in enumerate(self.state.objects_positions):
            if i == 4: break
            file_name = self.objects_names[i]
            for position in positions:
                i, j = position
                self.object_renders[position] = us.Entity(scale=(1/self.n, 1/self.n), model='quad', texture=f'images/{file_name}.png', origin_x=-.5, origin_y=.5, x=-.5 + (1/self.n)*i, y=.5 - (1/self.n)*j, z = -1, object_type = file_name)
    
    def step(self, action):
        if action not in range(5):
            raise TypeError("Action must be an integer in range [0,4]")
        
        carried_obj = np.argmax(self.state.carried_object)

        if carried_obj == 0:
            reward = -.4
        elif carried_obj == 1 or carried_obj == 3:
            reward = -.2
        else:
            reward = -1

        picked = False
        dropped = False
        item_dropped = -1

        if action in range(4):
            pos_before = self.state.ant_position
            pos = self.__move_ant(action)
            reward = -1 if pos_before == pos else reward
        elif action == 4:
            if self.state.carried_object[0] == 1:
                object_to_pick = self.__get_object_to_pick()

                if object_to_pick != -1: 
                    picked = True                 
                    self.__change_carried_object(object_to_pick)
                    reward = .3 if object_to_pick in [1,3] else -1
                else:   #tried to pick nothing                    
                    reward = -1
            else:
                if self.__ant_can_drop_object():
                    item_dropped = np.argmax(self.state.carried_object)
                    if item_dropped in [1,3]:
                        reward = 1 if self.state.ant_position in self.state.objects_positions[-1] else -1
                    else:
                        reward = -1 if self.state.ant_position in self.state.objects_positions[-1] else -.6 
                    dropped = True
                    self.__drop_object()
                else: #tried to do an illegal drop
                    reward = -1
            
        done = len(self.state.objects_positions[0]) == 0 and len(self.state.objects_positions[2]) == 0 and self.state.carried_object[0] == 1

        if done:
            reward = 10

        if self.render:
            self.__update(picked,dropped,item_dropped,action)

            self.app.step()
            self.app.step()

        return self.state, reward, done

    def __move_ant(self, action):
        x,y = self.state.ant_position
        if action == 0 and y > 0:
            self.state.ant_position = (x,y-1)
        if action == 1 and y < self.n-1:
            self.state.ant_position = (x,y+1)
        if action == 2 and x < self.n-1:
            self.state.ant_position = (x+1,y)
        if action == 3 and x > 0:
            self.state.ant_position = (x-1,y)

        return self.state.ant_position
                    
    def __get_object_to_pick(self):
        for i in range(4):
            if self.state.ant_position in self.state.objects_positions[i]:
                return i + 1
        return -1

    def __change_carried_object(self, obj_index):
        self.state.objects_positions[obj_index-1].remove(self.state.ant_position)
        self.state.carried_object[0] = 0
        self.state.carried_object[obj_index] = 1
    
    def __ant_can_drop_object(self):
        for i in range(len(self.state.objects_positions) - 1): #check every object but the anthill, cause you can actually drop an item over the anthill
            if self.state.ant_position in self.state.objects_positions[i]:
                return False
        return True

    def __drop_object(self):
        index = np.argmax(self.state.carried_object) # index of the current object that is being carried 
        if self.state.ant_position not in self.state.objects_positions[-1]:
            self.state.objects_positions[index-1].add(self.state.ant_position)
        self.state.carried_object[index] = 0
        self.state.carried_object[0] = 1
    
    def __update(self, picked, dropped, itemDropped,lastAction = -1):
            self.ant_render.x = -self.ant_offset + (1/self.n)*self.state.ant_position[0]
            self.ant_render.y = self.ant_offset - (1/self.n)*self.state.ant_position[1]

            rotations = {0 : 0, 1 : 180, 2 : 90, 3 : -90}

            if lastAction in range(4):
                self.ant_render.rotation_z = rotations[lastAction]
                   
            i = self.state.ant_position[0]
            j = self.state.ant_position[1]

            if picked:
                file_name = self.object_renders[(i,j)].object_type
                self.ant_render.texture = f'images/antwith{file_name}.png'
                self.object_renders[(i,j)].disable()
                _ = self.object_renders.pop((i,j))
            if dropped:
                self.ant_render.texture = 'images/ant.png'
                file_name = self.objects_names[itemDropped-1]
                if (i,j) not in self.state.objects_positions[-1]:
                    self.object_renders[(i,j)] = us.Entity(scale=(1/self.n, 1/self.n), model='quad', texture=f'images/{file_name}.png', origin_x=-.5, origin_y=.5, x=-.5 + (1/self.n)*i, y=.5 - (1/self.n)*j, z = -1, object_type = file_name)

    def reset(self):
        self.state = None
        self.state = self.__get_random_state()

        if self.render:
            self.__disable_renders()

            self.__buildWindow()

        return self.state

    def __disable_renders(self):
        for entity in self.object_renders.values():
            entity.disable()

        for item in self.ground_renders:
            item.disable()

        self.ground_renders.clear()

        self.object_renders.clear()

        self.ant_render.disable()

    def close(self):
        if self.render:
            self.app.destroy()
