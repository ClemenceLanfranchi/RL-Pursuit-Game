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

SARSA = "SARSA"
Q_LEARNING = "Q_learning"
EPSILON_GREEDY = "epsilon_greedy"
SOFTMAX = "softmax"

# Choose methods for learning and exploration
rl_algorithm = SARSA #Q_LEARNING
explore_method = EPSILON_GREEDY #SOFTMAX


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
    td = r + gamma * q[s_prime][a_max] - q[s][a]
    return q[s][a] + alpha * td

# Act with epsilon greedy
def act_with_epsilon_greedy(s, q):
    a = np.random.choice(np.argwhere(q[s]==np.max(q[s])).flatten())
    #print("a greedy", a)
    if np.random.rand() < epsilon:
        a = np.random.randint(5)
    return a

# Compute SARSA update
def sarsa_update(q,s,a,r,s_prime,a_prime):
    td = r + gamma * q[s_prime][a_prime] - q[s][a]
    return q[s][a] + alpha * td

def softmax(q):
    assert tau >= 0.0
    q_tilde = q - np.max(q)
    factors = np.exp(tau * q_tilde)
    return factors / np.sum(factors)

# Act with softmax
def act_with_softmax(s, q):
    prob_a = softmax(q[s, :])
    cumsum_a = np.cumsum(prob_a)
    return np.where(np.random.rand() < cumsum_a)[0][0]


# Evaluate a policy on n runs
def evaluate_policy(q,env,n,h):
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
    
    rewards_list = []
    successes = []
    for i_episode in range(n_episode):
        
        
        env.reset()
        states = visions(env)
        actions = []
         # Select the first action in this episode
        if explore_method == SOFTMAX:
            for i in range(env.nb_hunters):
                actions.append(act_with_softmax(states[i].tobytes(), q_table))
        elif explore_method == EPSILON_GREEDY:
            for i in range(env.nb_hunters):
                actions.append(act_with_epsilon_greedy(states[i].tobytes(), q_table))
        else:
            raise ValueError("Wrong Explore Method:".format(explore_method))
        
        
        images =[env.show()]
        rewards_episode = []
        
        for i_step in range(max_horizon):

            # Act
            obs_prime, rewards, done = env.step(actions)
            images.append(env.show())
            rewards_episode.append(sum(rewards))
            states_prime = visions(env)
            #total_return += np.power(gamma,i_step) *r
            

            # Select an action
            actions_prime = []
            if explore_method == SOFTMAX:
                for i in range(env.nb_hunters):
                    actions_prime.append(act_with_softmax(states_prime[i].tobytes(), q_table))
            elif explore_method == EPSILON_GREEDY:
                for i in range(env.nb_hunters):
                    actions_prime.append(act_with_epsilon_greedy(states[i].tobytes(), q_table))

            # Update a Q value table
            if rl_algorithm == SARSA:
                for i in range(env.nb_hunters):
                    q_table[states[i].tobytes()][actions[i]] = sarsa_update(q_table,states[i].tobytes(),actions[i],rewards[i],states_prime[i].tobytes(),actions_prime[i])

            elif rl_algorithm == Q_LEARNING:
                for i in range(env.nb_hunters):
                    q_table[states[i].tobytes()][actions[i]] = q_learning_update(q_table,states[i].tobytes(),actions[i],rewards[i],states_prime[i].tobytes())
            else:
                raise ValueError("Wrong RL algorithm:".format(rl_algorithm))
                
            # Transition to new state
            states = states_prime.copy()
            actions = actions_prime.copy()
            

            if done:
                successes.append(1)
        
                break
            if i_step == max_horizon-1 :
                successes.append(0)
                
        rewards_list.append(np.mean(rewards_episode))
        
        if i_episode%100==0:
            show_video(images, i_episode)
            epsilon = epsilon * epsilon_decay
        # Schedule for epsilon
        
        
        # Schedule for tau
        tau = init_tau + i_episode * tau_inc

    plt.figure(0)
    plt.plot([np.mean(rewards_list[i*100:(i+1)*100]) for i in range(n_episode//100)])
    #plt.title("Greedy policy with {0} and {1}".format(rl_algorithm))
    plt.xlabel("100 Steps")
    plt.ylabel("rewards")
    plt.show()
    
    plt.figure(1)
    plt.plot([np.mean(successes[i*100:(i+1)*100]) for i in range(n_episode//100)])
    #plt.title("Greedy policy with {0} and {1}".format(rl_algorithm))
    plt.xlabel("Steps")
    plt.ylabel("Success rate")
    plt.show()
        
if __name__ == "__main__":

    main()
    
    
