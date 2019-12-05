import numpy as np
import time
from pendulum import PendulumEnv
import random
import os

from math import isclose


class QLearning():

    def __init__(self, goal_theta_num=0, goal_theta_den=1):
        self.goal_theta_num = goal_theta_num
        self.goal_theta_den = goal_theta_den
        goal_theta = goal_theta_num / goal_theta_den

        self.env = PendulumEnv(goal_theta=goal_theta)

        self.save_directory = 'saved_policies'

        self.epsilon = 0
        self.gamma = .99

        self.num_avail_actions = 31
        self.num_fw_thdot = 31
        self.num_avail_positions = 51
        self.num_avail_velocities = 51

        self.thetas = np.linspace(-self.env.angle_limit, self.env.angle_limit, self.num_avail_positions)
        self.theta_dots = np.linspace(-self.env.max_speed, self.env.max_speed, self.num_avail_velocities)
        self.actions = np.linspace(-self.env.max_motor_speed, self.env.max_motor_speed, self.num_avail_actions)
        self.fwdot = np.linspace(-self.env.flywheel_max_thdot, self.env.flywheel_max_thdot, self.num_fw_thdot)

        self.q_matrix = np.zeros((
            self.num_avail_actions,
            self.num_avail_positions,
            self.num_avail_velocities,
            self.num_fw_thdot
        ))

        self.converge_threshold = 0.05 # % of q-value that dq must be close to to be considered converged
        self.perc_conv_thresh = 0.4 # % of q-values that must pass the convergence threshold

        self.percent_converged = 0
        self.percent_unexplored = 0

        self.prev_q_matrix = np.zeros((
            self.num_avail_actions,
            self.num_avail_positions,
            self.num_avail_velocities,
            self.num_fw_thdot
        )) # previous q-matrix

        self.dq_matrix = 100 * np.ones((
            self.num_avail_actions,
            self.num_avail_positions,
            self.num_avail_velocities,
            self.num_fw_thdot
        )) # delta-q matrix, tracks amount each weight is being updated
    

    def getQMatrixIdx(self, th, thdot, torque, fwdot):
        thIdx = np.abs(self.thetas - th).argmin()
        thdotIdx = np.abs(self.theta_dots - thdot).argmin()
        torIdx = np.abs(self.actions - torque).argmin()
        fwdotIdx = np.abs(self.fwdot - fwdot).argmin()

        return torIdx, thIdx, thdotIdx, fwdotIdx


    def getMaxQValue(self, th, thdot, fwdot):
        # returns the depth index for given th, thdot, fwdot state where torque is highest
        maxQValIdx = self.q_matrix[:, th, thdot, fwdot].argmax()
        maxQVal = self.q_matrix[maxQValIdx, th, thdot, fwdot]

        return maxQValIdx, maxQVal


    def get_action(self, th, thdot, fwdot):

        random_float = random.random()
        if (random_float < self.epsilon): # if a random float is less than our epsilon, explore a random action
            chosen_idx = np.random.randint(0, self.num_avail_actions)

        else: # if the random float is not less than epsilon, exploit the policy
            _, thIdx, thdotIdx, fwdotIdx = self.getQMatrixIdx(th, thdot, 0, fwdot)
            chosen_idx, _ = self.getMaxQValue(thIdx, thdotIdx, fwdotIdx)

        action = self.actions[chosen_idx]
        s = np.array([action]).astype(np.float32)

        return s
    

    def check_converged(self):
        # total number of elements
        total_elements = self.q_matrix.size
        
        percent_changed = np.divide(self.dq_matrix, self.q_matrix,
                        out=np.ones(self.q_matrix.shape),
                        where=self.q_matrix != 0)

        # compute the number of 'converged' q-values
        num_converged = (percent_changed < self.converge_threshold).sum()
        # percentage of converged q-values
        self.percent_converged = num_converged / total_elements

        self.percent_unexplored = (self.dq_matrix == 100).sum() / self.dq_matrix.size

        if self.percent_converged >= self.perc_conv_thresh:
            return True

        else:
            return False


    def train(self, episodes=50000, max_iterations=100000, l_rate=0.1):
        self.start_time = time.time()

        for episode_num in range(episodes):
            
            # reset the environment and declare th,thdot
            th, thdot, fwdot = self.env.reset()

            iter_count = -1

            while(not self.env.is_done and iter_count < max_iterations):
                iter_count += 1

                # select a new action to take
                u = self.get_action(th, thdot, fwdot)

                # find the current indecies in the self.weights_matrix so that we can update the weight for this action : th,thdot,u
                currTorIdx, currThIdx, currThdotIdx, currFwdotIdx = self.getQMatrixIdx(th, thdot, u, fwdot)

                # find next state corresponding to chosen action
                nextTh, nextThdot, nextFwdot, reward = self.env.step(u)

                _, nextThIdx, nextThdotIdx, nextFwdotIdx = self.getQMatrixIdx(nextTh, nextThdot, nextFwdot, u)

                # find the highest weighted torque in the self.weights_matrix given the nextTh,nextThdot
                _, nextQVal = self.getMaxQValue(nextThIdx, nextThdotIdx, nextFwdotIdx)

                self.q_matrix[currTorIdx, currThIdx, currThdotIdx, currFwdotIdx] += l_rate * (reward + self.gamma * nextQVal \
                                                                    - self.q_matrix[currTorIdx, currThIdx, currThdotIdx, currFwdotIdx])


                self.dq_matrix[currTorIdx, currThIdx, currThdotIdx, currFwdotIdx] = self.q_matrix[currTorIdx, currThIdx, currThdotIdx, currFwdotIdx] \
                                                                    - self.prev_q_matrix[currTorIdx, currThIdx, currThdotIdx, currFwdotIdx]

                self.prev_q_matrix[currTorIdx, currThIdx, currThdotIdx, currFwdotIdx] = self.q_matrix[currTorIdx, currThIdx, currThdotIdx, currFwdotIdx]

                th = nextTh
                thdot = nextThdot
                fwdot = nextFwdot
                    
                if iter_count % 100 == 0:
                    self.env.render()
                    print('iter_count = ', iter_count)
                    print('episode = ', episode_num)
                    print('epsilon = ', self.epsilon)
                    print(f'percent converged = {round(self.percent_converged, 2)}')
                    print(f'percent unexplored = {round(self.percent_unexplored, 2)}')
                    print("Time Elapsed: ",time.strftime("%H:%M:%S",time.gmtime(time.time()-self.start_time)))
                    print('')
                    
            # converged = self.check_converged()
            # if converged:
            #     print(f'Converged on episode {episode_num}')
            #     break
            
            # self.increase_epsilon_maybe(episode_num)
        
        self.print_stuff()

        self.save_policy()

    
    def increase_epsilon_maybe(self, ep_num):
        if ep_num % 1000 == 0 and ep_num != 0:
            self.epsilon -= .003

    
    def save_policy(self):
        time_struct = time.localtime(time.time())
        fname = self.get_fname(time_struct)
        save_path = os.path.join(self.save_directory, fname)
        np.save(save_path, self.q_matrix)

        print(f'saved policy: {fname}')


    def get_fname(self, time_params):
        year = time_params.tm_year
        month = time_params.tm_mon
        day = time_params.tm_mday
        hour = time_params.tm_hour
        minute = time_params.tm_min
        sec = time_params.tm_sec

        if isclose(self.goal_theta_num, np.pi, abs_tol=1e-7):
            num = 'pi'
        elif isclose(self.goal_theta_num, -np.pi, abs_tol=1e-7):
            num = 'ip'
        else:
            num = self.goal_theta_num

        if isclose(self.goal_theta_den, np.pi, abs_tol=1e-7):
            den = 'pi'
        elif isclose(self.goal_theta_den, -np.pi, abs_tol=1e-7):
            den = 'ip'
        else:
            den = self.goal_theta_den

        fname = f'{year}_{month}_{day}_{hour}_{minute}_{sec}_{num}_{den}'

        return fname
    

    def print_stuff(self):

        print("Training Done")
        print("Total Time Elapsed: ",time.strftime("%H:%M:%S",time.gmtime(time.time()-self.start_time)))
        print(f'percent converged = {round(self.percent_converged, 2)}')
        print(f'percent unexplored = {round(self.percent_unexplored, 2)}')