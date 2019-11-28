from qlearning import QLearning
from pendulum import PendulumEnv
import numpy as np
import time


def main():

    solution = QLearning()

    try:
        solution.train()
    
    except KeyboardInterrupt:
        pass
    
    solution.env.close()


def render_test():

    env = PendulumEnv()
    env.reset()

    try:
        for _ in range(500):
            env.render()
            # next two lines should be commented out to do the 'switcheroo'
            u = np.array([env.max_torque]).astype(env.action_space.dtype)
            info = env.step(u)

            # the next line should be commented out to do the 'switcheroo'
            # a = env.action_space.sample()
            # info = env.step(a)
            # print(a)

            time.sleep(.04)
    
    except KeyboardInterrupt:
        pass
    
    env.close()


if __name__ == '__main__':

    main()