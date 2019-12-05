import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import rendering


class PendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, goal_theta=0):
        self.max_speed = 8
        self.max_torque = 50 # max torque = 38.6
        self.dt = .05
        self.g = 9.81

        self.goal_theta = goal_theta

        self.is_done = False
        self.started_right = None
        self.switched_sides = False

        small_sprock_teeth = 6
        big_sprock_teeth = 60
        self.gear_ratio = small_sprock_teeth / big_sprock_teeth

        self.flywheel_diameter = .4
        self.max_motor_speed = 603.19 # rad/s
        self.flywheel_th = 0
        self.flywheel_max_thdot = self.max_motor_speed * self.gear_ratio
        self.flywheel_I = 54461568.26 / 1e9 # flywheel inertia in kg * m^2
        self.flywheel_prev_thdot = 0

        self.m_wheel = 2.43842
        motor_mass = .89
        self.m_pend = 9.54063 - self.m_wheel + motor_mass
        self.l = .68093
        self.angle_limit = np.pi/2 - np.arccos(self.l / (np.sqrt((self.flywheel_diameter/2)**2 + (self.l)**2)))

        self.state = None
        self.pend_prev_thdot = 0
        
        self.viewer = None
        self.rotation_add = 0

        self.seed()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, ms):
        th, thdot = self.state # th := theta
        dt = self.dt

        fwthdot = ms * self.gear_ratio
        fwthdot = np.clip(fwthdot, -self.flywheel_max_thdot, self.flywheel_max_thdot)

        pendulum_acc = (thdot - self.pend_prev_thdot) / dt
        flywheel_acc = (fwthdot - self.flywheel_prev_thdot) / dt

        u = self.flywheel_I * (pendulum_acc + flywheel_acc)
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering

        costs = self.calculate_cost(th, thdot, u)
        newthdot = self.calculate_new_thetadot(th, thdot, flywheel_acc, u)
        newth = self.calculate_new_theta(th, newthdot)

        self.state = np.array([newth, newthdot])
        
        self.pend_prev_thdot = thdot
        self.flywheel_prev_thdot = fwthdot

        self.flywheel_th += fwthdot * dt # for rendering

        self.check_if_done()

        return newth, newthdot, fwthdot, -costs

    
    def calculate_cost(self, theta, theta_dot, torque):
        # costs = theta**2 + .001 * theta_dot**2
        costs = (self.goal_theta - theta)**2 + .1*theta_dot**2 + .00001*(torque**2)

        return costs


    def calculate_new_thetadot(self, theta, theta_dot, fly_acc, torque):
        at_rest = self.is_at_rest(theta, torque)
        if at_rest:
            return 0

        if abs(theta) == self.angle_limit:
            theta_dot = 0

        g = self.g
        m_pend = self.m_pend
        m_wheel = self.m_wheel
        l = self.l
        dt = self.dt

        num = (m_pend*l/2 + m_wheel*l) * g * np.sin(theta) - self.flywheel_I * fly_acc
        den = (m_pend*l**2)/3 + m_wheel*l**2 + self.flywheel_I
        new_theta_dot = theta_dot + (num / den) * dt
        new_theta_dot = np.clip(new_theta_dot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        return new_theta_dot

    
    def is_at_rest(self, theta, torque):
        g = self.g
        m_pend = self.m_pend
        m_wheel = self.m_wheel
        l = self.l
        gravitational_torque = (m_pend*l/2 + m_wheel*l) * g * np.sin(theta)

        if theta == self.angle_limit and torque < gravitational_torque:
            return True

        elif theta == -self.angle_limit and -torque < abs(gravitational_torque):
            return True

        else:
            return False


    def calculate_new_theta(self, theta, theta_dot):
        dt = self.dt

        newth = theta + theta_dot * dt
        newth = np.clip(newth, -self.angle_limit, self.angle_limit)

        return newth

    
    def check_if_done(self):
        self.check_switched_sides()
        if self.switched_sides and abs(self.state[0]) == self.angle_limit:
            self.is_done = True
        

    def check_switched_sides(self):
        if self.started_right:
            if self.state[0] > 0:
                self.switched_sides = True
        
        else:
            if self.state[0] < 0:
                self.switched_sides = True


    def reset(self):
        self.state = np.array([
            np.random.uniform(-self.angle_limit, self.angle_limit, 1)[0],
            np.random.uniform(-self.max_speed, self.max_speed, 1)[0]
        ])
        self.flywheel_thdot = np.array([
            np.random.uniform(-self.flywheel_max_thdot, self.flywheel_max_thdot, 1)[0]
        ])

        self.is_done = False
        self.switched_sides = False

        if self.state[0] >= 0:
            self.started_right = False
        else:
            self.started_right = True

        self.last_u = None

        full_state = np.concatenate((self.state, self.flywheel_thdot))

        return full_state


    def render(self, mode='human'):
        l = self.l
        if self.viewer is None:
            
            self.viewer = rendering.Viewer(720,720)
            a = 1.8
            self.viewer.set_bounds(-(l*a),l*a,-(l*a),l*a)

            rod = rendering.make_capsule(l, l/15.2)
            rod.set_color(.04, .39, .12)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)

            axle = rendering.make_circle(.02)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)

            flywheel_diameter = self.flywheel_diameter
            flywheel_rim = rendering.make_circle(flywheel_diameter/2,filled=False)
            flywheel_rim.set_linewidth(7)
            flywheel_rim.set_color(0.5,0.5,0.5)
            self.flywheel_rim_transform = rendering.Transform()
            flywheel_rim.add_attr(self.flywheel_rim_transform)
            self.viewer.add_geom(flywheel_rim)

            flywheel_cross = rendering.make_cross(flywheel_diameter,l/20)
            flywheel_cross.set_color(0.5,0.5,0.5)
            self.flywheel_cross_transform = rendering.Transform()
            flywheel_cross.add_attr(self.flywheel_cross_transform)
            self.viewer.add_geom(flywheel_cross)

            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, .3, .3)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

            fname = path.join(path.dirname(__file__), "assets/gears.png")
            self.sprocket = rendering.Image(fname, 1, 1)
            self.sprocket_trans = rendering.Transform()
            self.sprocket.add_attr(self.sprocket_trans)
            self.viewer.add_geom(self.sprocket)
            self.sprocket_trans.scale = (self.flywheel_diameter/1.6,self.flywheel_diameter/1.8)

            st = (0, 0)
            le = 1.5
            end = (-le * np.sin(self.goal_theta), le * np.cos(self.goal_theta))
            # 0x0101 dotted
            # 0x00FF dashed
            # 0x1C47 dashed/dot/dash
            thickness = 7
            pattern = 0x1C47
            setpoint_line = rendering.Line(st, end, thickness, pattern)
            setpoint_line.set_color(0, 0, 0)
            self.viewer.add_geom(setpoint_line)

        self.viewer.add_onetime(self.img)

        theta = self.state[0] + np.pi/2
        self.pole_transform.set_rotation(theta)

        sprocket_theta_offset = np.pi / 180
        sprocket_length_offset = 0.05
        self.sprocket_trans.set_translation((l+sprocket_length_offset) * np.cos(theta - sprocket_theta_offset+0.012), 
                                            (l+sprocket_length_offset) * np.sin(theta - sprocket_theta_offset))
        self.sprocket_trans.set_rotation(theta + np.pi * (2.795))
        
        flywheel_offset = 0
        self.flywheel_rim_transform.set_translation(l * np.cos(theta - flywheel_offset), l * np.sin(theta - flywheel_offset))
        self.flywheel_rim_transform.set_rotation(theta + np.pi/4)

        self.flywheel_cross_transform.set_translation(l * np.cos(theta - flywheel_offset), l * np.sin(theta - flywheel_offset))
        self.flywheel_cross_transform.set_rotation(self.flywheel_th)


        img_offset = np.pi / 180
        img_length_offset = 0.15
        self.imgtrans.translation = ((l+img_length_offset) * np.cos(theta - img_offset+0.02), (l+img_length_offset) * np.sin(theta - img_offset))
        
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/(self.max_torque/2), np.abs(self.last_u)/(self.max_torque/2))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
