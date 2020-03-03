# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:19:23 2020
@author: Romain
"""
import numpy as np


class Environement:
    
    
    shape = None
    nb_hunters = 0
    preys = 0
    hunters = []
    positions = []
    actions = []
    step = 0
    
    action_to_delta = {
    0:np.array([0,1]),
    1:np.array([0,-1]),
    2:np.array([1,0]),
    3:np.array([-1,0]),
    4:np.array([0,0]) #stay still
    }
    
    
    def __init__(self,shape,nb_hunters = 4,positions = None, actions = list(range(4))):
        self.shape = shape
        self.nb_hunters = nb_hunters
        self.actions = actions
        
        if positions == None:
            #hunters are put at the four corners of the environment and the prey at the center
            positions = [[0,0],[0,self.shape],[self.shape,0],[self.shape, self.shape],[self.shape//2+1, self.shape//2+1]]
        
        self.hunters = []
        for i in range(nb_hunters):
            self.hunters.append(Agent(1,positions[i]))
        self.prey = Agent(0, positions[hunters+1])
                
    def voisins(self, position): #pour l'instant c'est omniscient 
        p = self.positions.copy()
        p.remove(position)
        return p

    def select_possible_actions(self, position): 
        p = []
        p.append(4)
        if position[0]>0:
            p.append(3)
        if position[0] < self.shape[0]:
            p.append(2)
        if position[1] >0:
            p.append(1)
        if position[1] < self.shape[1]:
            p.append(0)
        return p
            
    def step(self,actions):
        
        for i,a in enumerate(self.agents):
            voisins = self.voisins(a.position)
            possible_actions = self.select_possible_actions(a.position)
            action = a.decision(voisins,possible_actions)
            new_position = self.move(a,action)
            self.positions[i] = new_position
            
        for i in range(self.hunters):
            self.positions[i] += self.action_to_delta[actions[i]]
        
        self.step += 1
        return self.positions, self.done(), self.reward()
    
    
    def move_prey(self, p_still=0.5):
        u = np.random.rand()
        if u<p_still:
            return 4 #the prey remains still
        else:
            possible_actions = self.select_possible_actions(self.prey.position)
            return np.random.choice(possible_actions[1:]) #so that we don't consider standing still
        
    
    def reward(self):
        reward = 0
        for i in range(nb_hunters):
            if np.abs(self.hunters[i].position[0]-self.prey.position[0])
            +np.abs(self.hunters[i].poisition[1]-self.prey.position[1]) ==1: #the hunter is next to the prey
                reward+=10
        
        if reward == 40: #the 4 hunters have circled the prey
            return 100;
        
        nb_possible_actions_prey = len(self.select_possible_actions(self.prey.position))
        if reward == 30 and nb_possible_actions_prey==4: #there are 3 hunters around the prey + 1 wall
            return 100;
        
        if reward == 20 and nb_possible_actions_prey==3: #there are 2 huters around the prey + 2 walls
            return 100;
        
        return reward
    
    def done(self):
        if self.reward==100:
            return True
        return False

        

class Agent:
    
    role = None
    position = None
    decision_function = None
    
    def __init__(self,role,position,decision_function = lambda _,l : np.random.choice(l)):
        self.role = role
        self.position = position
        self.decision_function = decision_function
        
    def decision(self,voisions,possible_actions):
        return self.decision_function(voisins,possible_actions) # Par défaut c'est np.random.choice