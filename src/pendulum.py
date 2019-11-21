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

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 500.
        self.dt = .05
        self.g = g

        self.m_pend = 1.
        self.m_wheel = 1.
        self.l = 3

        self.flywheel_diameter = 1.7
        self.flywheel_ang_vel = 0
        self.flywheel_ang = 0
        self.flywheel_max_ang_vel = 15
        self.angle_limit = np.pi/2 - np.arccos(self.l/(np.sqrt((self.flywheel_diameter/2)**2 + (self.l)**2)))
        self.viewer = None
        self.rotation_add = 0

        high = np.array([1., 1., self.max_speed])
        # low = -self.max_torque
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        th, thdot = self.state # th := theta

        g = self.g
        m_pend = self.m_pend
        m_wheel = self.m_wheel
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        # m = 1.
        # newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        num = -(m_pend*l/2 + m_wheel*l) * g * np.sin(th + np.pi) - u
        den = (m_pend*l**2)/3 + m_wheel*l**2
        newthdot = thdot + (num / den) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111
        newth = np.clip(newth, -self.angle_limit, self.angle_limit)

        self.state = np.array([newth, newthdot])

        # make the flywheel speed up if given more torque
        newAngAcc = self.flywheel_ang_vel + (u*0.5)*dt
        self.flywheel_ang_vel = self.flywheel_ang_vel + (u*0.005)*dt
        self.flywheel_ang_vel = np.clip(self.flywheel_ang_vel, -self.flywheel_max_ang_vel, self.flywheel_max_ang_vel)
        newAng = self.flywheel_ang + self.flywheel_ang_vel * dt + 0.5*newAngAcc*dt**2
        newAngAcc = np.clip(newAngAcc,-self.flywheel_max_ang_vel, self.flywheel_max_ang_vel)
        print("new_Vel = ",self.flywheel_ang_vel)

        self.rotation_add = self.rotation_add + newAng #self.rotation_add + np.pi/10

        return self._get_obs(), -costs, False, {u}

    def reset(self):
        high = np.array([np.pi/2, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.state = [0, 0] #uncomment this and do the switcheroo in the main loop if you want the pendulum to be "frozen" -> (aayyy skrskr)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        l = self.l
        if self.viewer is None:
            self.viewer = rendering.Viewer(720,720)
            self.viewer.set_bounds(-5,5,-5,5)

            rod = rendering.make_capsule(l, .2)
            rod.set_color(.04, .39, .12)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)

            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)

            flywheel_diameter = self.flywheel_diameter
            flywheel_rim = rendering.make_circle(flywheel_diameter/2,filled=False)
            flywheel_rim.set_linewidth(7)
            flywheel_rim.set_color(0.5,0.5,0.5)
            self.flywheel_rim_transform = rendering.Transform()
            flywheel_rim.add_attr(self.flywheel_rim_transform)
            self.viewer.add_geom(flywheel_rim)

            flywheel_cross = rendering.make_cross(flywheel_diameter,0.1)
            flywheel_cross.set_color(0.5,0.5,0.5)
            self.flywheel_cross_transform = rendering.Transform()
            flywheel_cross.add_attr(self.flywheel_cross_transform)
            self.viewer.add_geom(flywheel_cross)

            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

            fname = path.join(path.dirname(__file__), "assets/gears.png")
            self.sprocket = rendering.Image(fname, 1, 1)
            self.sprocket_trans = rendering.Transform()
            self.sprocket.add_attr(self.sprocket_trans)
            self.viewer.add_geom(self.sprocket)
            self.sprocket_trans.scale = (self.flywheel_diameter/1.6,self.flywheel_diameter/1.8)

            # uncomment if you want to have an axle at the opposite end of the rod
            # need to comment out equal lines above
            #flywheel = rendering.make_circle(0.85,filled=False)
            #flywheel.set_linewidth(5)
            #flywheel.set_color(0.5,0.5,0.5)
            #self.flywheel_transform = rendering.Transform()
            #flywheel.add_attr(self.flywheel_transform)
            #self.viewer.add_geom(flywheel)             

        self.viewer.add_onetime(self.img)

        theta = self.state[0] + np.pi/2
        self.pole_transform.set_rotation(theta)

        sprocket_theta_offset = np.pi / 180
        sprocket_length_offset = 0.21
        self.sprocket_trans.set_translation((l+sprocket_length_offset) * np.cos(theta - sprocket_theta_offset+0.012), 
                                            (l+sprocket_length_offset) * np.sin(theta - sprocket_theta_offset))
        self.sprocket_trans.set_rotation(theta + np.pi * (2.795))
        
        flywheel_offset = 0
        self.flywheel_rim_transform.set_translation(l * np.cos(theta - flywheel_offset), l * np.sin(theta - flywheel_offset))
        self.flywheel_rim_transform.set_rotation(theta + np.pi/4)

        self.flywheel_cross_transform.set_translation(l * np.cos(theta - flywheel_offset), l * np.sin(theta - flywheel_offset))
        self.flywheel_cross_transform.set_rotation(theta + self.rotation_add)


        img_offset = np.pi / 180
        img_length_offset = 0.68
        self.imgtrans.translation = ((l+img_length_offset) * np.cos(theta - img_offset+0.02), (l+img_length_offset) * np.sin(theta - img_offset))
        
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/(self.max_torque/2), np.abs(self.last_u)/(self.max_torque/2))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
