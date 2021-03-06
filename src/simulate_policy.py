from pendulum import PendulumEnv
from qlearning import QLearning

import numpy as np
import time
import os


class Simulator():

    def __init__(self, data_dictionary=None, policy_name=None, policy_directory=None):

        self.save_directory = 'rory_data'

        self.trainer = QLearning()
        self.data = data_dictionary

        self.num_actions = self.trainer.num_avail_actions
        self.num_positions = self.trainer.num_avail_positions
        self.num_velocities = self.trainer.num_avail_velocities

        if data_dictionary is None:
            if policy_directory is None:
                self.load_directory = self.trainer.save_directory
            else:
                self.load_directory = policy_directory

            if policy_name is None:
                self.policy_name = self.grab_newest_policy()
            else:
                self.policy_name = policy_name

            # goal_theta_num, goal_theta_den = self.get_goal_theta(self.policy_name)

            # self.goal_theta = goal_theta_num / goal_theta_den
            self.goal_theta = np.pi / 4

            self.file = os.path.join(self.load_directory, self.policy_name)
            self.policy = self.load_policy()
            self.data = dict()
            

        else:
            self.goal_theta = self.data['goal_theta']
            self.policy = self.data['policy']
            self.file = self.data['fname']
            self.policy_name = self.file

        self.env = PendulumEnv(goal_theta=self.goal_theta)

        self.thetas = np.linspace(-self.env.angle_limit, self.env.angle_limit, self.num_positions)
        self.theta_dots = np.linspace(-self.env.max_speed, self.env.max_speed, self.num_velocities)
        self.actions = np.linspace(-self.env.max_torque, self.env.max_torque, self.num_actions)

        self.dummy_q = self.trainer.q_matrix
        self.num_useful_actions = 0

        self.torques = []
        self.theta_errors = []

    
    def load_policy(self):
        policy = np.load(self.file, allow_pickle=True)
        policy = policy.item().get('policy')
        return policy

    
    def grab_newest_policy(self):
        all_policies = os.listdir(self.load_directory)

        file_count = 0
        theta_dict = {}
        name_dict = {}

        for i, policy in enumerate(all_policies):

            if not policy.startswith('.'): # ignore .DS files, its a Mac thing
                fname, _ = os.path.splitext(policy)
                
                try:
                    gtheta = '_'.join(fname.split('_')[-2:])
                    fname = '_'.join(fname.split('_')[:-2])
                    time_components = np.array(list(map(int, fname.split('_'))))
                    file_count += 1
                except ValueError:
                    continue

                theta_dict[i] = gtheta

                if file_count == 1:
                    name_array = time_components
                else:
                    name_array = np.row_stack((name_array, time_components))

                name_dict[fname] = i

        while len(name_array.shape) > 1:
            col_idx_diff =  np.any(name_array != name_array[0,:], axis = 0).nonzero()[0][0]
            row_idx_curr_max = np.argwhere(name_array[:, col_idx_diff] == np.amax(name_array[:, col_idx_diff])).squeeze()
            name_array = name_array[row_idx_curr_max, :]
        
        newest_policy = name_array
        newest_policy = '_'.join(map(str, newest_policy))
        suffix_theta = theta_dict[name_dict[newest_policy]]
        newest_policy += '_' + suffix_theta + '.npy'
            
        return newest_policy
    

    def get_goal_theta(self,pol_name):
        # same exact logic as get_newest_policy() except it just grabs the goal theta
        fname, _ = os.path.splitext(pol_name)
        len_fname = len(fname) - 1

        if 'pi' in fname:
            idx_pi = fname.find('pi')
            
            if idx_pi != len_fname: # index of numerator
                num = np.pi
            else: # index of denominator
                den = np.pi

            fname = fname.replace('pi','555')
        
        if 'ip' in fname:
            idx_ip = fname.find('ip')

            if idx_ip != len_fname: # index of numerator
                num = -np.pi
            else: # index of denominator
                den = -np.pi

            fname = fname.replace('ip','555')
                       
        time_components = np.array(list(map(int, fname.split('_'))))
        name_array = time_components

        while len(name_array.shape) > 1:
            col_idx_diff =  np.any(name_array != name_array[0,:], axis = 0).nonzero()[0][0]
            row_idx_curr_max = np.argwhere(name_array[:, col_idx_diff] == np.amax(name_array[:, col_idx_diff])).squeeze()
            name_array = name_array[row_idx_curr_max, :]
        
        newest_policy = name_array
        temp_num = newest_policy[-2]
        temp_den = newest_policy[-1]

        if temp_num != 555:
            num = temp_num
        if temp_den != 555:
            den = temp_den

        return num, den


    def getQMatrixIdx(self, th, thdot, torque):
        thIdx = np.abs(self.thetas - th).argmin()
        thdotIdx = np.abs(self.theta_dots - thdot).argmin()
        torIdx = np.abs(self.actions - torque).argmin()

        return torIdx, thIdx, thdotIdx


    def getMaxQValue(self, th, thdot):
        # returns the depth index for given th,thdot state where torque is highest
        maxQValIdx = self.policy[:, th, thdot].argmax()
        maxQVal = self.policy[maxQValIdx, th, thdot]

        return maxQValIdx, maxQVal


    def get_action(self, th, thdot):
        _, thIdx, thdotIdx = self.getQMatrixIdx(th, thdot, 0)
        chosen_idx, _ = self.getMaxQValue(thIdx, thdotIdx)

        action = self.actions[chosen_idx]
        u = np.array([action]).astype(self.env.action_space.dtype)

        return u

    
    def save_precious_simulated_data(self):
        self.data["simulated_episodes"] = self.num_episodes
        self.data["simulated_iterations"] = self.num_iterations
        self.data["avg_sim_cost"] = self.avg_cost
        self.data["perc_useful_actions"] = self.num_useful_actions
        self.data["torque_arr"] = self.torques
        self.data["theta_error_arr"] = self.theta_errors

        fname = self.policy_name

        save_path = os.path.join(self.save_directory, fname)

        if not os.path.exists(self.save_directory):
            os.mkdir(self.save_directory)

        np.save(save_path, self.data)

        print(f'saved precious data: {fname}')


    def update_dummy_q(self,torI,thI,thDotI):
        if self.dummy_q[torI,thI,thDotI] != 1000:
            self.dummy_q[torI,thI,thDotI] = 1000

    
    def simulate(self, ep_num=500, iter_num=150, start_pos=None, start_vel=None):
        print(f'Running simulation using policy: {self.file}')

        self.num_episodes = ep_num
        self.num_iterations = iter_num
        total_total_cost = 0

        try:

            for i in range(self.num_episodes):

                th, thdot = self.env.reset()

                if start_pos is not None:
                    self.env.state[0] = start_pos
                    th = start_pos
                
                if start_vel is not None:
                    self.env.state[1] = start_vel
                    thdot = start_vel

                for _ in range(self.num_iterations):

                    self.env.render()

                    self.theta_errors.append(self.goal_theta - th)

                    u = self.get_action(th, thdot)
                    self.torques.append(u)

                    torIdx,thIdx,thDotIdx = self.getQMatrixIdx(th, thdot, u)

                    self.update_dummy_q(torIdx,thIdx,thDotIdx)

                    nextTh, nextThdot, reward = self.env.step(u)
                    total_total_cost += reward

                    th = nextTh
                    thdot = nextThdot
                
                    time.sleep(.05)

                if i != self.num_episodes-1:
                    self.torques = []
                    self.theta_errors = []
        
        except KeyboardInterrupt:
            pass

        self.avg_cost = total_total_cost / self.num_episodes
        self.num_useful_actions = (self.dummy_q == 1000).sum() / self.dummy_q.size
        
        self.env.close()


def main():
    dummy_env = PendulumEnv()
    start_pos = dummy_env.angle_limit

    pol_dir = 'saved_policies'
    fname = 'interp316_pi84.npy'
    sim = Simulator(policy_directory=pol_dir, policy_name=fname)
    sim.simulate(ep_num=1, iter_num=200, start_pos=start_pos, start_vel=0)
    sim.save_precious_simulated_data()


if __name__ == '__main__':
    main()