#from qlearning import QLearning
#from pendulum import PendulumEnv
#from simulate_policy import Simulator
import numpy as np
import os
import matplotlib.pyplot as plt

'''
The last two numbers in a given filename are the numerator and denominator of the goal_theta for the policy
If a set point was at -pi/2, then the filename stores this as 'ip' to distinguish between the positive pi noted as 'pi'
E.x. the file '2019_12_9_0_32_41_ip_8.npy' would have a goal_theta of -pi/8
     the file '2019_12_9_0_32_41_pi_4.npy' would have a goal_theta of pi/4
     the file '2019_12_9_0_32_41_0_1.npy' would have a goal_theta of 0 (vertical)

Each .npy file in 'precious_data' is a dictionary (d) that contains:

d["gamma"] (float) --> weight for future reward (kept constant at 0.99 for all policies)
d["epsilon"] (float) --> threshold for choosing random actions 
d["goal_theta"] (float) --> setpoint for pendulum to balace at
d["perc_actions_explored"] (float) --> % of total array that had the q-value change during training
d["perc_converged"] (float) --> % of total array that met the following convergence criteria
d["conv_perc_req"] (float) --> % of q-values that must pass the convergence threshold
d["converge_threshold"] (float) --> % of q-value that dq must be close to to be considered converged
d["training_time"] (float) --> did not actually sucessfully store the time taken for training :(
d["training_episodes"] (int) --> number of training episodes
d["training_iterations"] (int) --> max number of iterations/episode
d["learning_rate"] (float) --> learning rate (kept constant at 0.1 for all policies)
d["policy"] (matrix) --> policy learned during training
d["simulated_episodes"] (int) --> number of episodes for simulation
d["simulated_iterations"] (int) --> number of iterations/episode for simulation
d["avg_sim_cost"] (float) --> average cost calculated over the simulation (lower cost means better policy)
d["perc_useful_actions"] (float) --> percentage of actions that were used during the simulation to get the pendulum to balance over 500 random initial conditions
d["fname"] (str) --> name of the file
d["converged_stats"] (dict) --> dictionary where the keys are threshold values (from [0-0.95] in 0.05 step size) and 
                                the values are the percentage of the q-matrix that meets the threshold of convergence

ONLY FOR THE FOLLOWING 7 POLICIES:
 - 2019_12_10_5_44_2_0_1
 - 2019_12_10_4_8_13_ip_3
 - 2019_12_10_4_12_1_pi_3
 - 2019_12_10_4_9_3_ip_4
 - 2019_12_10_4_21_6_pi_4
 - 2019_12_10_4_35_13_ip_8
 - 2019_12_10_4_35_20_pi_8

d["total_reward_arr"] (list) --> list of total reward every 100 episodes over 100,000 episode training session


THE FOLLOWING POLICIES WERE RUN FOR 100k EPISODES AT 1k OR 2k ITERATIONS, WHERE 'epsilon' WAS CHANGED FROM 0.2 TO 0.1 AT EPISODE 50k
(this was done to see if this lower epsilon encouraged 'perc_converged' to be a higher value
 - 2019_12_10_0_10_39_0_1
 - 2019_12_10_2_45_1_0_1
 - 2019_12_9_23_18_9_pi_4
 - 2019_12_10_0_39_41_pi_4
 - 2019_12_10_2_15_44_ip_4
 - 2019_12_10_3_54_13_ip_4


 THE FOLLOWING POLICIES WERE RUN FOR 100k EPISODES AT 1k ITERATIONS, WHERE 'epsilon' WAS CHANGED FROM 0.2 TO 0.3 AT EPISODE 50k
(this was done to see if this higher epsilon encouraged 'perc_converged' to be a higher value
 - 2019_12_10_5_26_53_0_1
 - 2019_12_10_5_3_21_pi_4
 - 2019_12_10_4_59_19_ip_4
'''

def load_policy(file):
    policy = np.load(file, allow_pickle=True)
    return policy

def get_dict_val(dict, key):
    '''
    once dictionary is loaded, this will retrieve the value associated with the given key
    Example:
    pol_dict = load_policy('2019_12_9_0_32_41_pi_8.npy') # loads the policy dictionary
    pol = get_dict_val(pol_dict,'policy') # loads the policy associated with '2019_12_9_0_32_41_pi_8.npy' into pol
    useful_actions = get_dict_val(pol_dict,'perc_useful_actions') # loads the value of percent useful actions into useful_actions
    '''
    value = dict.item().get(key) 
    return value

def main():
    load_directory = 'precious_data'
    policy_name = '2019_12_10_5_44_2_0_1.npy' # one of the policies that has Total reward array
    file = os.path.join(load_directory, policy_name)
    pol_dict1 = load_policy(file)
    print(pol_dict1)

    #all_policy_files = os.listdir(load_directory)
    #for policy_file in all_policy_files:
    #    policy_dict = load_policy('2019_12_9_0_32_41_pi_8.npy')

    
    
if __name__ == '__main__':
    main()


