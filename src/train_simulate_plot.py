from qlearning import QLearning
from pendulum import PendulumEnv
from simulate_policy import Simulator
import numpy as np
import time


def main():
    desired_th_num = 0
    desired_th_den = 1
    tr_ep = 10000 # number of training episodes
    tr_it = 150000   # number of training iterations per episode
    solution = QLearning(goal_theta_num = desired_th_num, goal_theta_den = desired_th_den)
    start_pos = solution.env.angle_limit
    start_vel = 0

    try:
        solution.train(episodes=tr_ep, max_iterations=tr_it)
    
    except KeyboardInterrupt:
        solution.save_policy()
        solution.get_precious_data()
    
    solution.env.close()

    data_dict = solution.data

    sim = Simulator(data_dictionary = data_dict)
    try:
        sim.simulate(ep_num=2, iter_num=200, start_pos=start_pos, start_vel=start_vel)
    except KeyboardInterrupt:
        pass
    sim.save_precious_simulated_data()
    

if __name__ == '__main__':
    main()

    # torque_type = 0
    # render_test(torque_type)