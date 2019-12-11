from qlearning import QLearning
from pendulum import PendulumEnv
from simulate_policy import Simulator
import numpy as np
import time


def main():
    pend = PendulumEnv()
    desired_th_num = 0
    desired_th_den = 1
    start_pos = pend.angle_limit
    start_vel = 0
    tr_ep = 1000 # number of training episodes
    tr_it = 100000   # number of training iterations per episode
    solution = QLearning(goal_theta_num = desired_th_num, goal_theta_den = desired_th_den)

    try:
        solution.train(episodes=tr_ep, max_iterations=tr_it, start_pos=start_pos, start_vel=start_vel)
    
    except KeyboardInterrupt:
        solution.save_policy()
        solution.get_precious_data()
    
    solution.env.close()

    data_dict = solution.data

    sim = Simulator(data_dictionary = data_dict)
    try:
        sim.simulate(ep_num=1, iter_num=200, start_pos=start_pos, start_vel=start_vel)
    except KeyboardInterrupt:
        pass
    sim.save_precious_simulated_data()

    # #TODO
    # data_dict = sim.data
    

if __name__ == '__main__':
    main()

    # torque_type = 0
    # render_test(torque_type)