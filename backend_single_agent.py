# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:19:23 2020
@author: Romain
"""
import numpy as np


class Environement:
    
    
    shape = 15
    vision = 2
    hunters = 0
    preys = 0
    agents = []
    positions = []
    actions = []
    step = 0
    
    action_to_delta = {
    0:np.array([0,1]),
    1:np.array([0,-1]),
    2:np.array([1,0]),
    3:np.array([-1,0]),
    4:np.array([0,0])
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
        self.prey = Agent(0, positions[self.nb_hunters+1])
    
    def move_prey(self, p_still=0.5):
        u = np.random.rand()
        if u<p_still:
            return 4 #the prey remains still
        else:
            possible_actions = self.select_possible_actions(self.prey.position)
            return np.random.choice(possible_actions[1:]) #so that we don't consider standing still
        
                
    def position_other_hunters(self, position): #pour l'instant c'est omniscient 
        p = self.positions.copy()
        p.remove(position)
        return p

    def select_possible_actions(self, position): 
        p = []
        p.append(4)
        if position[0]>0 :
            p.append(3)
        if position[0] < self.shape[0] :
            p.append(2)
        if position[1] > 0 :
            p.append(1)
        if position[1] < self.shape[1] :
            p.append(0)
        return p
            
    def step(self,actions):
        
        pos_prey= self.prey.position + self.action_to_delta[self.move_prey()]
        pos_hunters = []
        
        for i in range(self.nb_hunters):
            a = self.hunters[i]
            possible_actions = self.select_possible_actions(a.position)
            if actions[i] in possible_actions :
                pos_hunters.append(  self.hunters[i].position + self.action_to_delta[actions[i]])
        
        moving = [True, True , True, True, True]
        for i in range(self.nb_hunters):
            if pos_prey == pos_hunters[i]:
                moving[0]=False
                moving[i+1] = False
            for j in range(i+1,self.nb_hunters):
                if pos_hunters[i] == pos_hunters[j] :
                    moving[i+1]=False
                    moving[j+1] = False
        
        if moving[0] :
            self.prey.position = pos_prey
        for i in range(self.nb_hunters):
            if moving[i+1] :
                self.hunters[i].position = pos_hunters[i]
        
        self.step += 1
        
        return self.visions(), self.done(), self.reward()
        
    
    
    def surrounding_state(self,hunter):
        vision = self.vision
        shape = self.shape
        state = []
        pos_hunter = self.hunters[hunter].position
        pos_prey = self.prey.position
        if np.abs(pos_prey[0]-pos_hunter[0])<=vision and np.abs(pos_prey[1]-pos_hunter[1])<=vision :
            state.append(pos_prey - pos_hunter)
        else : 
            state.append(np.array([-1,-1]))
        for i in range (self.nb_hunters):
            pos_other_hunter = self.hunters[i].position
            if i != hunter and np.abs(pos_other_hunter[0]-pos_hunter[0])<=vision and np.abs(pos_other_hunter[1]-pos_hunter[1])<=vision  :
                state.append(pos_other_hunter - pos_hunter)
            else : 
                state.append(np.array([-1,-1]))
        if pos_hunter[0]<vision :
            pos_wall_x = -1-pos_hunter[0]
        elif pos_hunter[0]>=shape - vision :
            pos_wall_x = shape-pos_hunter[0]
        else :
            pos_wall_x = -1
        if pos_hunter[1]<vision :
            pos_wall_y = -1-pos_hunter[1]
        elif pos_hunter[1]>=shape - vision :
            pos_wall_y = shape-pos_hunter[1]
        else :
            pos_wall_y = -1    
        
        state.append(np.array([pos_wall_x,pos_wall_y]))
        return state
    
    def visions(self):
        visions = []
        for i in range(self.nb_hunters):
            visions.append(self.surrounding_state(i))
        return visions
        
    def reward(self):
        reward = 0
        for i in range(self.nb_hunters):
            if np.abs(self.hunters[i].position[0]-self.prey.position[0])+np.abs(self.hunters[i].poisition[1]-self.prey.position[1]) ==1: #the hunter is next to the prey
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
        return self.decision_function(voisions,possible_actions) # Par défaut c'est np.random.choice