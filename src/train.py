import numpy as np
import time
import sys
from pendulum import PendulumEnv
import random
import copy

class QLearning():
    def __init__(self):
        self.env = PendulumEnv()
        self.env.reset()
        self.action_number = 101
        self.state_space = 2
        self.epsilon = 0.2
        self.bias = 0
        self.gamma = 0.99
        min_torque = -self.env.max_torque
        max_torque = self.env.max_torque
        self.actions = []
        for i in range(self.action_number):
            self.actions.append(min_torque + (max_torque - min_torque) / (self.action_number - 1) * i)
        print (self.actions)
        self.weight_matrix = np.zeros((len(self.actions), self.state_space))

    def train(self, episodes=10, max_iterarions=20000, learning_rate=0.01):
        for episodes_number in range(episodes):
            total_reward = 0
            curr_state_info = self.env.reset()
            curr_state = np.zeros(self.state_space)
            for i in range(self.state_space):
                curr_state[i] = curr_state_info[i]
            
            #print (curr_state)
            for iteration_number in range(max_iterarions):
                time.sleep(0.1)
                print (curr_state)
                random_float = random.random()
                if (random_float < self.epsilon):
                    action = np.random.randint(0, self.action_number - 1)
                else:
                    action = -1
                    max_q_s_a_w = -sys.float_info.max
                    for action_iter in range(self.action_number):
                        q_s_a_w_iter = np.dot(curr_state, self.weight_matrix[action_iter]) + self.bias
                        if (q_s_a_w_iter > max_q_s_a_w):
                            max_q_s_a_w = q_s_a_w_iter
                            action = action_iter
                
                u = np.array([self.actions[action]]).astype(self.env.action_space.dtype)
                #u = np.array([500]).astype(self.env.action_space.dtype)
                
                next_state_info, reward, isDone, _ = self.env.step(u)
                print ("reward : ", reward)
                print (self.actions[action])
                print ("")

                next_state = np.zeros((self.state_space))
                for i in range(self.state_space):
                    next_state[i] = float(next_state_info[i])

                max_q_s_a_w = -sys.float_info.max
                for action_iter in range(self.action_number):
                    q_s_a_w_iter = np.dot(next_state, self.weight_matrix[action_iter]) + self.bias
                    if (q_s_a_w_iter > max_q_s_a_w):
                        max_q_s_a_w = q_s_a_w_iter

                gradient_matrix_w = np.zeros((self.action_number, self.state_space))
                gradient_matrix_w[action] = curr_state

                copy_of_weight_matrix = copy.deepcopy(self.weight_matrix)
                copy_of_bias = copy.deepcopy(self.bias)


                self.weight_matrix = self.weight_matrix - learning_rate * ((np.dot(curr_state, copy_of_weight_matrix[action]) + copy_of_bias) - (reward + self.gamma * max_q_s_a_w)) * gradient_matrix_w
                self.bias = self.bias - learning_rate * ((np.dot(curr_state, copy_of_weight_matrix[action]) + copy_of_bias) - (reward + self.gamma * max_q_s_a_w)) * 1.0;


                curr_state = next_state
                total_reward += reward
            
                if (True or episodes_number % 100 == 0):
                    self.env.render()




def main():
    solution = QLearning()
    solution.train()
    return 

    env = PendulumEnv()
    env.reset()

    try:
        for _ in range(500):
            env.render()
            # next two lines should be commented out to do the 'switcheroo'
            # u = np.array([0]).astype(env.action_space.dtype)
            # info = env.step(u)

            # the next line should be commented out to do the 'switcheroo'
            a = env.action_space.sample()
            info = env.step(a)
            print(a)

            time.sleep(.04)
    
    except KeyboardInterrupt:
        pass
    
    env.close()


if __name__ == '__main__':

    main()