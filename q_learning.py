# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:29:16 2020

@author: cleme
"""

import numpy as np
from collections import defaultdict 
import matplotlib.pyplot as plt
from backend_single_agent_v3 import Environment
from visualization import ImageResult, show_video
from collections import deque

 
# Meta parameters for the RL agent
alpha = 0.1
tau = init_tau = 1
tau_inc = 0.01
gamma = 0.99
epsilon = 0.5
epsilon_decay = 0.999

def surrounding_state(env, hunter):
    #positions of the hunters and prey and walls if they are visible
    #hunter : index of hunter in list of hunters
    vision = env.vision
    shape = env.shape
    state = []
    pos_hunter = env.hunters[hunter].position
    pos_prey = env.prey.position
    
    #position of the prey
    if np.abs(pos_prey[0]-pos_hunter[0])<=vision and np.abs(pos_prey[1]-pos_hunter[1])<=vision :
        state.append(pos_prey - pos_hunter)
    else : 
        state.append(np.array([-100,-100]))
        
    #position of the hunters
    nbh_vision = 0
    relative_positions=[]
    for i in range (env.nb_hunters):
        if i == hunter:
            break
        pos_other_hunter = env.hunters[i].position
        if np.abs(pos_other_hunter[0]-pos_hunter[0])<=vision and np.abs(pos_other_hunter[1]-pos_hunter[1])<=vision  :
            relative_positions.append(pos_other_hunter - pos_hunter)
            nbh_vision +=1
            
    np.sort(relative_positions, axis=0) #so that the state is not dependent on the order of the hunters
    for i in range(nbh_vision):
        state.append(relative_positions[i])
    
    for i in range(env.nb_hunters-nbh_vision-1):
        state.append(np.array([-100, -100]))
            
    #position of the walls       
    if pos_hunter[0]<vision :
        pos_wall_x = -1-pos_hunter[0]
    elif pos_hunter[0]>=shape - vision :
        pos_wall_x = shape-pos_hunter[0]
    else :
        pos_wall_x = -100
    if pos_hunter[1]<vision :
        pos_wall_y = -1-pos_hunter[1]
    elif pos_hunter[1]>=shape - vision :
        pos_wall_y = shape-pos_hunter[1]
    else :
        pos_wall_y = -100    
    
    state.append(np.array([pos_wall_x,pos_wall_y]))
    return state

def visions(env):
    #array of the surrounding states of all hunters
    visions = []
    for i in range(env.nb_hunters):
        visions.append(surrounding_state(env,i))
    return np.array(visions)

# Compute Q-Learning update
def q_learning_update(q,s,a,r,s_prime):
    #print("q_sprime",q[s_prime, :])
    #print("a", a)
    a_max = np.argmax(q[s_prime])
    td = r + gamma * q[s_prime, a_max] - q[s, a]
    return q[s,a] + alpha * td

# Act with epsilon greedy
def act_with_epsilon_greedy(s, q):
    a = np.random.choice(np.argwhere(q[s]==np.max(q[s])).flatten())
    #print("a greedy", a)
    if np.random.rand() < epsilon:
        a = np.random.randint(5)
    return a

# Evaluate a policy on n runs
def evaluate_policy(q,env,n,h,explore_type):
    success_rate = 0.0
    mean_return = 0.0

    for i in range(n):
        discounted_return = 0.0
        s = env.reset()

        for step in range(h):
            s,r, done = env.step(act_with_epsilon_greedy(s,q))
            discounted_return += np.power(gamma,step) * r

            if done:
                success_rate += float(r)/n
                mean_return += float(discounted_return)/n
                break
    return success_rate, mean_return

def main():
    global epsilon
    global tau
    
    env = Environment()
    
    n_a = 5
    
    # Init Q-table as a nested dictionary that maps state -> (action -> action-value). 
    q_table = defaultdict(lambda: np.zeros(n_a))
    
    # Experimental setup
    n_episode = 1000
    print("n_episode ", n_episode)
    max_horizon = 100
    eval_steps = 10
    
    # Monitoring perfomance
    window = deque(maxlen=100)
    last_100 = 0

    greedy_success_rate_monitor = np.zeros([n_episode,1])
    greedy_discounted_return_monitor = np.zeros([n_episode,1])

    behaviour_success_rate_monitor = np.zeros([n_episode,1])
    behaviour_discounted_return_monitor = np.zeros([n_episode,1])
    
    for i_episode in range(n_episode):
        
        total_return = 0.0
        
        env.reset()
        states = visions(env)
        actions = []
        for i in range(env.nb_hunters):
            actions.append(act_with_epsilon_greedy(states[i].tobytes(), q_table))
        
        images =[env.show()]
        
        for i_step in range(max_horizon):

            # Act
            obs_prime, r, done = env.step(actions)
            images.append(env.show())
            
            states_prime = visions(env)
            total_return += np.power(gamma,i_step) *r
            

            # Select an action
            actions_prime = []
            for i in range(env.nb_hunters):
                actions_prime.append(act_with_epsilon_greedy(states_prime[i].tobytes(), q_table))


            # Update a Q value table
            for i in range(env.nb_hunters):
                q_table[states[i].tobytes(), actions[i]] = q_learning_update(q_table,states[i].tobytes(),actions[i],r,states_prime[i].tobytes())

            # Transition to new state
            states = states_prime.copy()
            actions = actions_prime.copy()

            if done:
                window.append(r)
                last_100 = window.count(1)

                greedy_success_rate_monitor[i_episode-1,0], greedy_discounted_return_monitor[i_episode-1,0]= evaluate_policy(q_table,env,eval_steps,max_horizon,GREEDY)
                behaviour_success_rate_monitor[i_episode-1,0], behaviour_discounted_return_monitor[i_episode-1,0] = evaluate_policy(q_table,env,eval_steps,max_horizon,explore_method)
                if verbose:
                    print("Episode: {0}\t Num_Steps: {1:>4}\tTotal_Return: {2:>5.2f}\tFinal_Reward: {3}\tEpsilon: {4:.3f}\tSuccess Rate: {5:.3f}\tLast_100: {6}".format(i_episode, i_step, total_return, r, epsilon,greedy_success_rate_monitor[i_episode-1,0],last_100))
                    #print "Episode: {0}\t Num_Steps: {1:>4}\tTotal_Return: {2:>5.2f}\tTermR: {3}\ttau: {4:.3f}".format(i_episode, i_step, total_return, r, tau)

                break
            
        if i_episode%100==0:
            show_video(images, i_episode)

        # Schedule for epsilon
        epsilon = epsilon * epsilon_decay
        # Schedule for tau
        tau = init_tau + i_episode * tau_inc

    plt.figure(0)
    plt.plot(range(0,n_episode,10),greedy_success_rate_monitor[0::10,0])
    plt.title("Greedy policy with {0} and {1}".format(rl_algorithm,explore_method))
    plt.xlabel("Steps")
    plt.ylabel("Success Rate")

    plt.figure(1)
    plt.plot(range(0,n_episode,10),behaviour_success_rate_monitor[0::10,0])
    plt.title("Behaviour policy with {0} and {1}".format(rl_algorithm,explore_method))
    plt.xlabel("Steps")
    plt.ylabel("Success Rate")
    plt.show()
        
if __name__ == "__main__":

    main()
        
        