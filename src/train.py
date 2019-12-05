from qlearning import QLearning
from pendulum import PendulumEnv
import numpy as np
import time


def render_test(torque_type=0):
    '''
    torque_types:
        - 0 : square wave torque to make the pendulum oscillate back and forth
        - 1 : some constant value torque
        - 2 (or anything else) : random torque
    '''

    env = PendulumEnv()
    a = env.reset()

    at_rest = False
    val = 0
    
    try:
        for _ in range(500):
            env.render()
            
            if torque_type == 0:

                if env.state[0] == env.angle_limit and at_rest:
                    val = env.flywheel_max_thdot
                    at_rest = False

                elif env.state[0] == -env.angle_limit and at_rest:
                    val = -env.flywheel_max_thdot
                    at_rest = False
                
                if abs(env.state[0]) == env.angle_limit and not at_rest:
                    at_rest = True

                s = np.array([val]).astype(np.float32)

            elif torque_type == 1:
                val = 0
                s = np.array([val]).astype(np.float32)

            else:
                val = np.array([
                    np.random.uniform(-env.flywheel_max_thdot, env.flywheel_max_thdot, 1)[0]
                ])
                s = np.array([val]).astype(np.float32)

            
            info = env.step(s)
            
            time.sleep(.1)
    
    except KeyboardInterrupt:
        pass
    
    env.close()


def main():

    theta_num = 0
    theta_den = 1
    solution = QLearning(goal_theta_num=theta_num, goal_theta_den=theta_den)

    try:
        solution.train()
    
    except KeyboardInterrupt:
        solution.save_policy()
        print(f'percent converged = {round(solution.percent_converged, 2)}')
        print(f'percent unexplored = {round(solution.percent_unexplored, 2)}')
    
    solution.env.close()


if __name__ == '__main__':
    main()

    # torque_type = 1
    # render_test(torque_type)