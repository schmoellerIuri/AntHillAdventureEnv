# -*- coding: utf-8 -*-
"""
MDP Solution using value iteration algorithm for the environment shown in the book: 

Artificial Intelligence A Modern Approach - Third Edition (Russel,Norvig - 2010; P. 645)

State numbers:

    8  9 10 11
    4  -  6  7
    0  1  2  3

@author: Iuri Schmoeller
"""

import numpy as np

def getTransition(state, action):
    return transitions[state][action]

def getReward(state):
    if state == goal:
        return 1
    elif state == secondGoal:
        return -1
    else:
        return -0.04
    
def valueIteration(states, actions, probs, discountFactor = 1):
    utilities = np.array([0,0,0,0,0,0,0,-1,0,0,0,1], dtype='float64')

    j = 0

    while j < 20:
        utilitiesCopy = np.copy(utilities)
        for state in states:
            if state == goal or state == secondGoal or state == 5:
                continue

            probUtilitySums = []
            for i in range(len(actions)):
                left = (i - 1) % len(actions)
                right = (i + 1) % len(actions)

                soma = 0

                if(getTransition(state, actions[i]) != -1):
                    soma += probs[0]*utilities[getTransition(state, actions[i])]
                if(getTransition(state, actions[left]) != -1):
                    soma += probs[1]*utilities[getTransition(state, actions[left])]    
                if(getTransition(state, actions[right]) != -1):
                    soma += probs[1]*utilities[getTransition(state, actions[right])]

                probUtilitySums.append(soma)
            
            utilitiesCopy[state] = getReward(state) + discountFactor * max(probUtilitySums)
    
        utilities = np.copy(utilitiesCopy)
        j += 1

    return utilities
    

actions = np.array([0, 1, 2, 3])  # 0: up, 1: right, 2: down, 3: left

env = np.array(np.arange(12))

goal = 11
secondGoal = 7


# Transition model for the env. - Ex: At State 0 if the agent takes the action 0(up) it goes to state 4
transitions = np.array([[4, 1, 0, 0],[1, 2, 1, 0],[6, 3, 2, 1],[7, 3, 3, 2],[8, 4, 0, 4],[-1, -1, -1, -1],[10, 7, 2, 6],[-1, -1, -1, -1],[8, 9, 4, 8],[9, 10, 9, 8],[10, 11, 6, 9],[-1, -1, -1, -1]])

probs = np.array([0.8,0.1,0.1])

utilities = valueIteration(env, actions, probs)

map_actions = {0: '↑', 1: '→', 2: '↓', 3: '←'}

for i in range(12):
    if i == goal or i == secondGoal or i == 5:
        continue
    
    probUtilitySums = []
    for a in range(len(actions)):
        left = (a - 1) % len(actions)
        right = (a + 1) % len(actions)

        soma = 0

        if(getTransition(i, a) != -1):
            soma += probs[0]*utilities[getTransition(i, a)]
        if(getTransition(i, left) != -1):
            soma += probs[1]*utilities[getTransition(i, left)]    
        if(getTransition(i, right) != -1):
            soma += probs[1]*utilities[getTransition(i, right)]

        probUtilitySums.append(soma)
    
    print('State:', i, 'Action:', map_actions[np.argmax(probUtilitySums)]) #Optimal policy