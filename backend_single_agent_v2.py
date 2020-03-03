# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:19:23 2020
@author: Romain
"""
import numpy as np
from visualization import ImageResult, show_video
import time

class Environement:
    
    
    shape = 15
    vision = 2
    hunters = 0
    preys = 0
    agents = []
    actions = []
    step_nb = 0
    
    action_to_delta = {
    0:np.array([0,1]),
    1:np.array([0,-1]),
    2:np.array([1,0]),
    3:np.array([-1,0]),
    4:np.array([0,0])
    }
    
    def __init__(self,shape=15,nb_hunters = 4,positions = None, actions = list(range(4))):
        self.shape = shape
        self.nb_hunters = nb_hunters
        self.actions = actions
        self.step_nb = 0
        
        if positions == None:
            #hunters are put at the four corners of the environment and the prey at the center
            positions = [[0,0],[0,self.shape-1],[self.shape-1,0],[self.shape-1, self.shape-1],[self.shape//2, self.shape//2]]
        
        self.hunters = []
        for i in range(nb_hunters):
            self.hunters.append(Agent(1,positions[i]))
        self.prey = Agent(0, positions[self.nb_hunters])
    
    def move_prey(self, p_still=0.5):
        u = np.random.rand()
        if u<p_still:
            return 4 #the prey remains still
        else:
            possible_actions = self.select_possible_actions(self.prey.position)
            return np.random.choice(possible_actions[1:]) #so that we don't consider standing still
        

    def select_possible_actions(self, position): 
        p = []
        p.append(4)
        if position[0]>0 :
            p.append(3)
        if position[0] < self.shape -1 :
            p.append(2)
        if position[1] > 0 :
            p.append(1)
        if position[1] < self.shape -1:
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
            else : 
                pos_hunters.append (np.array(self.hunters[i].position))
                
        
        
        #check for incompatible behaviour
        moving = [True, True , True, True, True]
        for i in range(self.nb_hunters):
            if pos_prey[0] == pos_hunters[i][0] and pos_prey[1] == pos_hunters[i][1]:
                moving[0]=False
                moving[i+1] = False
            for j in range(i+1,self.nb_hunters):
                if pos_hunters[i][0] == pos_hunters[j][0] and pos_hunters[i][1] == pos_hunters[j][1] :
                    moving[i+1]=False
                    moving[j+1] = False
        
        #update the agents positions if needed
        if moving[0] :
            self.prey.position = pos_prey
        for i in range(self.nb_hunters):
            if moving[i+1] :
                self.hunters[i].position = pos_hunters[i]
        
        self.step_nb += 1
        
        return self.get_all_positions(), self.done(), self.reward()
        
    def get_all_positions(self) :
        positions = [self.prey.position]
        for i in range(self.nb_hunters):
            positions.append(self.hunters[i].position)
        return np.array(positions)
    
    def surrounding_state(self,hunter):
        #positions of the hunters and prey and walls if they are visible
        #hunter : index of hunter in list of hunters
        vision = self.vision
        shape = self.shape
        state = []
        pos_hunter = self.hunters[hunter].position
        pos_prey = self.prey.position
        
        #position of the prey
        if np.abs(pos_prey[0]-pos_hunter[0])<=vision and np.abs(pos_prey[1]-pos_hunter[1])<=vision :
            state.append(pos_prey - pos_hunter)
        else : 
            state.append(np.array([None,None]))
            
        #position of the hunters
        for i in range (self.nb_hunters):
            pos_other_hunter = self.hunters[i].position
            if i != hunter and np.abs(pos_other_hunter[0]-pos_hunter[0])<=vision and np.abs(pos_other_hunter[1]-pos_hunter[1])<=vision  :
                state.append(pos_other_hunter - pos_hunter)
            else : 
                state.append(np.array([None,None]))
                
        #position of the walls       
        if pos_hunter[0]<vision :
            pos_wall_x = -1-pos_hunter[0]
        elif pos_hunter[0]>=shape - vision :
            pos_wall_x = shape-pos_hunter[0]
        else :
            pos_wall_x = None
        if pos_hunter[1]<vision :
            pos_wall_y = -1-pos_hunter[1]
        elif pos_hunter[1]>=shape - vision :
            pos_wall_y = shape-pos_hunter[1]
        else :
            pos_wall_y = None    
        
        state.append(np.array([pos_wall_x,pos_wall_y]))
        return state
    
    def visions(self):
        #array of the surrounding states of all hunters
        visions = []
        for i in range(self.nb_hunters):
            visions.append(self.surrounding_state(i))
        return np.array(visions)
        
    def reward(self):
        reward = 0
        for i in range(self.nb_hunters):
            if np.abs(self.hunters[i].position[0]-self.prey.position[0])+np.abs(self.hunters[i].position[1]-self.prey.position[1]) ==1: #the hunter is next to the prey
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

    def show(self):
        picture = ImageResult(self.shape,30,self.vision)
        return picture.draw_obs(self.get_all_positions())
        #picture.show()

        

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
    

def demo() :
    env = Environement()
    images =[env.show()]
    for i in range(100):
        env.step(np.random.randint(5, size=4))
        images.append(env.show())
    show_video(images)
    return

demo()
    