import time
import numpy as np
import ursina as us
import keyboard

class AntEnvironment():
    def __init__(self, render_gui=False) :
        self.render_gui = render_gui

        self.stateSpace = np.zeros(shape=(1,128), dtype=int)

        self.actionSpace = np.array(np.arange(0, 6))

        self.desc = np.array([["--------"], ["-x--x---"], ["x----xx-"],["L--x----"],["--xxx--S"], ["-x--x---"], ["--------"], ["-x----x-"]])
        self.desc = np.array([list(row[0]) for row in self.desc])
         
        self.currentDesc = np.copy(self.desc)

        self.antPosition = (4,7)

        self.leafPosition = 0

        self.done = False

        self.rendered = False

        self.winningStates = []

        self.losingStates = []

        self.transisitions = np.zeros(shape=(128, 6), dtype=int)
        self.rewards = np.zeros(shape=(128, 6), dtype=int)

        self.currentState = self.encode(4, 7, 0)

        for antrow in range(8):
            for antcol in range(8):
                for leafpos in range(2):
                    for action in range(6):
                        state = self.encode(antrow, antcol, leafpos)
                        newRow, newCol, newLeafPos = antrow, antcol, leafpos
                        reward = 0 if leafpos == 1 else -1
                    
                        if action == 0 and antrow > 0:
                            newRow = antrow - 1
                        elif action == 1 and antrow < 7:
                            newRow = antrow + 1
                        elif action == 2 and antcol < 7:
                            newCol = antcol + 1
                        elif action == 3 and antcol > 0:
                            newCol = antcol - 1
                        elif action == 4:
                            reward = -5
                            if leafpos == 0 and antrow == 3 and antcol == 0:
                                newLeafPos = 1
                                reward = 10
                        elif action == 5:
                            reward = -5
                            if leafpos == 1 and antrow == 4 and antcol == 7:
                                newLeafPos = 0
                                self.winningStates.append(self.encode(newRow, newCol, newLeafPos))
                                reward = 50
                        
                        newState = self.encode(newRow, newCol, newLeafPos)
                        
                        if self.desc[newRow, newCol] == 'x':
                            self.losingStates.append(newState)
                            reward = -10

                        self.transisitions[state, action] = newState
                        self.rewards[state, action] = reward

        if self.render_gui:
            self.buildWindow()
                        
    def close(self):
        if self.render_gui:
            self.app.destroy()
        

    def step(self, action):
        self.currentDesc[self.antPosition] = self.desc[self.antPosition]
        self.currentDesc[3,0] = 'L' if self.leafPosition == 0 else '-'
        newState = self.transisitions[self.currentState, action]
        self.antPosition = self.decode(newState)[0:2]
        self.leafPosition = self.decode(newState)[-1]
        self.currentDesc[self.antPosition] = 'a' if self.leafPosition == 0 else 'A'
        
        reward = self.rewards[self.currentState, action]
        self.done = newState in self.winningStates or newState in self.losingStates
        self.currentState = newState

        if self.render_gui:
            self.update(action)
        
            self.app.step()
            self.app.step()
        else:
            print(self.currentDesc)

        return newState, reward, self.done
    
    def buildWindow(self):
        self.app = us.Ursina()

        us.window.size = (800,800)
        us.window.color = us.color.rgb(102/255, 51/255, 0)
        us.camera.orthographic = True
        us.camera.fov = 1

        for i in range(8):
            for j in range(8):
                texture = 'images/ground.jpg'

                if (i,j) == (4,7):
                    texture = 'images/anthill.jpg'

                us.Entity(scale=(1/8, 1/8), model='quad', texture=texture, origin_x=0, origin_y=0, x=-.438 + 1/8*j, y=.438 - 1/8*i, z = 0)

                if self.desc[i,j] == 'x':
                    us.Entity(scale=(1/10, 1/10), model='quad', texture='images/poison.png', origin_x=0, origin_y=0, x=-.438 + 1/8*j, y=.438 - 1/8*i, z = -1)
                if (i,j) == self.antPosition:
                    self.antRender = us.Entity(scale=(1/10, 1/10), model='quad', texture='images/ant.png', origin_x=0, origin_y=0, x=-.438 + 1/8*j, y=.438 - 1/8*i, z = -1)
                if (i,j) == (3,0):
                    self.leafRender = us.Entity(scale=(1/15, 1/15), model='quad', texture='images/leaf.png', origin_x=0, origin_y=0, x=-.438 + 1/8*j, y=.438 - 1/8*i, z = -1)

        self.rendered = True
        for _ in range(10):
            self.app.step()
    
    def update(self, lastAction = -1):
            self.antRender.x = -.438 + 1/8*self.antPosition[1]
            self.antRender.y = .438 - 1/8*self.antPosition[0]

            if lastAction == 0:
                self.antRender.rotation_z = 0
            elif lastAction == 1:
                self.antRender.rotation_z = 180
            elif lastAction == 2:
                self.antRender.rotation_z = 90
            elif lastAction == 3:
                self.antRender.rotation_z = -90

            if self.leafPosition == 1:
                self.leafRender.disable()
                self.antRender.texture = 'images/antwithleaf.png'
            else:
                self.leafRender.enable()
                self.antRender.texture = 'images/ant.png'
    
    def reset(self):
        self.currentDesc = np.copy(self.desc)
        self.antPosition = (4,7)
        self.leafPosition = 0
        self.currentState = self.encode(4, 7, 0)
        self.done = False

        time.sleep(0.15)
        self.update()
        self.app.step()
        self.app.step()

        return self.currentState


    def encode(self, row, col, leaf):
        encoded = ((row * 8) + col)* 2 + leaf
        return encoded
    
    def decode(self, encoded):
        leaf = encoded % 2
        encoded = encoded // 2
        col = encoded % 8
        row = encoded // 8
        return row, col, leaf
        

ant = AntEnvironment(render_gui=True)

map_keys = {'w':0, 's':1, 'd':2, 'a':3, 'e':4, 'q':5, 'r':-1}

action = 0
acum_reward = 0
while action != -1:
    key = keyboard.read_key()
    if key in map_keys:
        action = map_keys[key]
        state, reward, done = ant.step(action)
        acum_reward += reward
        time.sleep(0.15)
        if done:
            print('Game Over')
            print('Acumulated Reward:', acum_reward)
            acum_reward = 0
            state = ant.reset()
    
ant.close()