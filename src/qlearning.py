import numpy as np
import time
import sys
from pendulum import PendulumEnv
import random
import copy
import os


class QLearning():

    def __init__(self):

        self.env = PendulumEnv()
        self.save_directory = 'saved_policies'

        self.epsilon = .2
        self.gamma = .95

        self.num_avail_actions = 31
        self.num_avail_positions = 51
        self.num_avail_velocities = 51

        self.thetas = np.linspace(-self.env.angle_limit, self.env.angle_limit, self.num_avail_positions)
        self.theta_dots = np.linspace(-self.env.max_speed, self.env.max_speed, self.num_avail_velocities)
        self.actions = np.linspace(-self.env.max_torque, self.env.max_torque, self.num_avail_actions)

        self.q_matrix = np.zeros((
            self.num_avail_actions,
            self.num_avail_positions,
            self.num_avail_velocities
        ))
    

    def getQMatrixIdx(self, th, thdot, torque):
        thIdx = np.abs(self.thetas - th).argmin()
        thdotIdx = np.abs(self.theta_dots - thdot).argmin()
        torIdx = np.abs(self.actions - torque).argmin()

        return torIdx, thIdx, thdotIdx


    def getMaxQValue(self, th, thdot):
        # returns the depth index for given th,thdot state where torque is highest
        maxQValIdx = self.q_matrix[:, th, thdot].argmax()
        maxQVal = self.q_matrix[maxQValIdx, th, thdot]

        return maxQValIdx, maxQVal


    def get_action(self, th, thdot):

        random_float = random.random()
        if (random_float < self.epsilon): # if a random float is less than our epsilon, explore a random action
            chosen_idx = np.random.randint(0, self.num_avail_actions)

        else: # if the random float is not less than epsilon, exploit the policy
            _, thIdx, thdotIdx = self.getQMatrixIdx(th, thdot, 0)
            chosen_idx, _ = self.getMaxQValue(thIdx, thdotIdx)

        action = self.actions[chosen_idx]
        u = np.array([action]).astype(self.env.action_space.dtype)

        return u


    def train(self, episodes=10000, max_iterations=150000, l_rate=0.1):
        self.start_time = time.time()

        for episode_num in range(episodes):
            
            # reset the environment and declare th,thdot
            th, thdot = self.env.reset()

            iter_count = -1

            while(not self.env.is_done and iter_count < max_iterations):
                iter_count += 1

                # select a new action to take
                u = self.get_action(th, thdot)

                # find the current indecies in the self.weights_matrix so that we can update the weight for this action : th,thdot,u
                currTorIdx, currThIdx, currThdotIdx = self.getQMatrixIdx(th, thdot, u)

                # find next state corresponding to chosen action
                nextTh, nextThdot, reward = self.env.step(u)

                _, nextThIdx, nextThdotIdx = self.getQMatrixIdx(nextTh, nextThdot, u)

                # find the highest weighted torque in the self.weights_matrix given the nextTh,nextThdot
                _, nextQVal = self.getMaxQValue(nextThIdx, nextThdotIdx)

                self.q_matrix[currTorIdx, currThIdx, currThdotIdx] = self.q_matrix[currTorIdx, currThIdx, currThdotIdx] \
                                                                        + l_rate * (reward + self.gamma * nextQVal \
                                                                        - self.q_matrix[currTorIdx, currThIdx, currThdotIdx])

                if iter_count % 100 == 0:
                    self.env.render()
                    print('iter_count = ', iter_count)
                    print('episode = ', episode_num)
                    print("Time Elapsed: ",time.strftime("%H:%M:%S",time.gmtime(time.time()-self.start_time)))
                    print('')

                th = nextTh
                thdot = nextThdot

        print("Training Done")
        print("Total Time Elapsed: ",time.strftime("%H:%M:%S",time.gmtime(time.time()-self.start_time)))

        self.save_policy()

    
    def save_policy(self):
        time_struct = time.localtime(time.time())
        fname = self.get_fname(time_struct)
        save_path = os.path.join(self.save_directory, fname)
        np.save(save_path, self.q_matrix)


    def get_fname(self, time_params):
        year = time_params.tm_year
        month = time_params.tm_mon
        day = time_params.tm_mday
        hour = time_params.tm_hour
        minute = time_params.tm_min
        sec = time_params.tm_sec

        fname = f'{year}_{month}_{day}_{hour}_{minute}_{sec}'

        return fname