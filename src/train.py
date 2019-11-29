from qlearning import QLearning
from pendulum import PendulumEnv
import numpy as np
import time


def render_test():

    env = PendulumEnv()
    env.reset()

    constant_torque = True
    at_rest = True

    try:
        for _ in range(500):
            env.render()
            
            if constant_torque:

                if env.state[0] == env.angle_limit and at_rest:
                    val = env.max_torque
                    at_rest = False

                elif env.state[0] == -env.angle_limit and at_rest:
                    val = -env.max_torque
                    at_rest = False
                
                if abs(env.state[0]) == env.angle_limit and not at_rest:
                    at_rest = True

                u = np.array([val]).astype(env.action_space.dtype)

            else:
                u = env.action_space.sample()
            
            info = env.step(u)
            # print(info)
            # print(env.state[0]) # print angular position
            print(env.state[1]) # print angular velocity
            
            time.sleep(.1)
    
    except KeyboardInterrupt:
        pass
    
    env.close()


def main():

    solution = QLearning()

    try:
        solution.train()
    
    except KeyboardInterrupt:
        pass
    
    solution.env.close()


if __name__ == '__main__':

    # main()
    render_test()