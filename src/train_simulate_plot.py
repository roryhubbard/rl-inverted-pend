from qlearning import QLearning
from pendulum import PendulumEnv
from simulate_policy import Simulator
import numpy as np
import time


def main():

    desired_th_num = 0
    desired_th_den = 1
    tr_ep = 100000 # number of training episodes
    tr_it = 2000   # number of training iterations per episode
    solution = QLearning(goal_theta_num = desired_th_num, goal_theta_den = desired_th_den)

    try:
        solution.train(episodes=tr_ep, max_iterations=tr_it)
    
    except KeyboardInterrupt:
        solution.save_policy()
        #print(f'percent converged = {round(solution.percent_converged, 2)}')
        #print(f'percent unexplored = {round(solution.percent_unexplored, 2)}')

    data_dict = solution.data
    solution.env.close()

    sim = Simulator(data_dictionary = data_dict)
    sim.simulate()

    #TODO
    data_dict = sim.data
    

if __name__ == '__main__':
    main()

    # torque_type = 0
    # render_test(torque_type)