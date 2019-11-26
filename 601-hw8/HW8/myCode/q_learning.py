from environment import MountainCar
import numpy as np
import sys
import random
import time
import copy


def main(args):
	mode = args[1]
	weight_out_filename = args[2]
	returns_out_filename = args[3]
	episodes = int(args[4])
	max_iterations = int(args[5])
	epsilon = float(args[6])
	gamma = float(args[7])
	learning_rate = float(args[8])
	
	env = MountainCar(mode)

	bias = 0
	weight_matrix = np.zeros((3, env.state_space))
	#print weight_matrix
	returns_out = []


	for episode_number in range(episodes):
		total_reward = 0
		curr_state_info = env.reset()
		curr_state = np.zeros(env.state_space)
		for key in curr_state_info:
			curr_state[key] = curr_state_info[key]

		for iteration_number in range(max_iterations):
			#choose an action
			#print curr_state
			#time.sleep(0.1)

			random_float = random.random()
			action = -1
			
			if (random_float < epsilon):
				#print "random"
				action = np.random.randint(0,3)
			else:
				#print "greedy"
				action = -1
				max_q_s_a_w = -sys.float_info.max
				for action_iter in range(3):
					q_s_a_w_iter = np.dot(curr_state, weight_matrix[action_iter]) + bias
					if (q_s_a_w_iter > max_q_s_a_w):
						max_q_s_a_w = q_s_a_w_iter
						action = action_iter

			# print "action : ", action


			next_state_info, reward, isDone = env.step(action)
			next_state = np.zeros((env.state_space))

			for key in next_state_info:
				next_state[key] = float(next_state_info[key])		
	
			# print "current state : ", curr_state
			# print "next state : ", next_state

			#update weight_matrix and bias
			max_q_s_a_w = -sys.float_info.max
			for action_iter in range(3):
				q_s_a_w_iter = np.dot(next_state, weight_matrix[action_iter]) + bias
				if (q_s_a_w_iter > max_q_s_a_w):
					max_q_s_a_w = q_s_a_w_iter
			
			

			gradient_matrix_w = np.zeros((3, env.state_space))
			gradient_matrix_w[action] = curr_state
			
			copy_of_weight_matrix = copy.deepcopy(weight_matrix)
			copy_of_bias = copy.deepcopy(bias)


			weight_matrix = weight_matrix - learning_rate * ((np.dot(curr_state, copy_of_weight_matrix[action]) + copy_of_bias) - (reward + gamma * max_q_s_a_w)) * gradient_matrix_w
			bias = bias - learning_rate * ((np.dot(curr_state, copy_of_weight_matrix[action]) + copy_of_bias) - (reward + gamma * max_q_s_a_w)) * 1.0;


			curr_state = next_state
			total_reward += reward
			if (isDone):
				break

		returns_out.append(total_reward)
		

	f = open(weight_out_filename, 'w')
	f.write(str(bias) + '\n')
	for i in range(len(weight_matrix[0])):
		for j in range(len(weight_matrix)):
			f.write(str(weight_matrix[j][i]) + '\n')
	f.close()

	f = open(returns_out_filename, 'w')
	for i in returns_out:
		f.write(str(i) + '\n') 
	f.close()



if __name__ == "__main__":
	main(sys.argv)