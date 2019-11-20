import numpy as np
import time
import sys
from pendulum import PendulumEnv


def main():
    env = PendulumEnv()
    env.reset()

    try:
        for _ in range(500):
            env.render()
            # next two lines should be commented out to do the 'switcheroo'
            #u = np.array([0]).astype(env.action_space.dtype)
            #info = env.step(u)

            # the next line should be commented out to do the 'switcheroo'
            info = env.step(env.action_space.sample())
            print(info)

            time.sleep(.04)
    
    except KeyboardInterrupt:
        pass
    
    env.close()


if __name__ == '__main__':

    main()