from qlearning import QLearning
from pendulum import PendulumEnv
import numpy as np
import time


def render_test():

    env = PendulumEnv()
    env.reset()

    constant_torque = False

    try:
        for _ in range(500):
            env.render()
            
            if constant_torque:
                val = env.max_torque
                u = np.array([val]).astype(env.action_space.dtype)
                info = env.step(u)

            else:
                u = env.action_space.sample()
                info = env.step(u)
            
            print(info)
            
            time.sleep(.04)
    
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

    main()