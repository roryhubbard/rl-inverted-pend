from pendulum import PendulumEnv
from qlearning import QLearning

import numpy as np
import time
import os


class Simulator():

    def __init__(self, policy_name=None):

        self.env = PendulumEnv()
        self.trainer = QLearning()
        self.num_actions = self.trainer.num_avail_actions
        self.num_positions = self.trainer.num_avail_positions
        self.num_velocities = self.trainer.num_avail_velocities

        self.load_directory = self.trainer.save_directory

        if policy_name is None:
            policy_name = self.grab_newest_policy()

        self.file = os.path.join(self.load_directory, policy_name)
        self.policy = self.load_policy()

        self.thetas = np.linspace(-self.env.angle_limit, self.env.angle_limit, self.num_positions)
        self.theta_dots = np.linspace(-self.env.max_speed, self.env.max_speed, self.num_velocities)
        self.actions = np.linspace(-self.env.max_torque, self.env.max_torque, self.num_actions)

    
    def load_policy(self):
        policy = np.load(self.file)
        return policy

    
    def grab_newest_policy(self):
        all_policies = os.listdir(self.load_directory)

        file_count = 0
        for policy in all_policies:
            if not policy.startswith('.'): # ignore .DS files, its a Mac thing
                file_count += 1
                fname, _ = os.path.splitext(policy)

                try:
                    time_components = np.array(list(map(int, fname.split('_'))))
                except ValueError:
                    continue

                if file_count == 1:
                    name_array = time_components
                else:
                    name_array = np.row_stack((name_array, time_components))
        
        while len(name_array.shape) > 1:
            col_idx_diff =  np.any(name_array != name_array[0,:], axis = 0).nonzero()[0][0]
            row_idx_curr_max = np.argwhere(name_array[:, col_idx_diff] == np.amax(name_array[:, col_idx_diff])).squeeze()
            name_array = name_array[row_idx_curr_max, :]
        
        newest_policy = name_array
        newest_policy = '_'.join(map(str, newest_policy)) + '.npy'
            
        return newest_policy


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

    
    def simulate(self):
        print(f'Running simulation using policy: {self.file}')

        th, thdot = self.env.reset()

        try:
            for _ in range(500):

                self.env.render()

                u = self.get_action(th, thdot)
                nextTh, nextThdot, _ = self.env.step(u)

                th = nextTh
                thdot = nextThdot
                
                time.sleep(.05)
        
        except KeyboardInterrupt:
            pass

        self.env.close()


def main():
    # fname = '2019_12_1_18_24_43.npy'
    sim = Simulator()
    sim.simulate()


if __name__ == '__main__':
    main()