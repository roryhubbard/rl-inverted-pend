import numpy as np 
import matplotlib.pyplot as plt
import pathlib as pb
from collections import OrderedDict


class Plotter():

    def __init__(self, load_dir, save_dir, save_fname):
        self.load_dir = pb.Path.cwd() / load_dir

        self.save_dir = pb.Path.cwd() / save_dir
        self.save_path = self.save_dir / save_fname

        self.data = None

        plt.rcParams['savefig.facecolor'] = 'xkcd:black'
        # plt.rcParams['figure.titlesize'] = 'medium'
        # plt.rcParams['axes.titlesize'] = 'medium'


    def load_data(self, fname=None):
        load_file = self.load_dir / fname
        self.data = np.load(load_file, allow_pickle=True)


    def get_item(self, key):
        value = self.data.item().get(key)
        return value

    
    def prepare_figure(self, fig, title=None, height=1.0, tight=True):
        self.fig = fig
        fig.patch.set_facecolor('xkcd:black')
        fig.suptitle(title, color='xkcd:white', y=height)
        if tight:
            fig.set_tight_layout(True)

    
    def prepare_axis(self, axis, title=None, xlabel=None, ylabel=None):
        axis.set_facecolor('xkcd:black')
        axis.tick_params(color='xkcd:white', labelcolor='xkcd:white')
        axis.spines['bottom'].set_color('xkcd:white')
        axis.spines['top'].set_color('xkcd:white')
        axis.spines['right'].set_color('xkcd:white')
        axis.spines['left'].set_color('xkcd:white')

        axis.set_title(title, color='xkcd:white')
        axis.set_xlabel(xlabel, color='xkcd:white')
        axis.set_ylabel(ylabel, color='xkcd:white')


    def plot(self, axis, x, y, label=None):
        if x is None:
            x = np.linspace(1, len(y), len(y))

        axis.plot(x, y, label=label)


    def make_legend(self, ax, outside=False):
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))

        if outside:
            ax.legend(by_label.values(), by_label.keys(),
                loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.legend(by_label.values(), by_label.keys(),
                loc='best')


    def save(self):
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
        
        self.fig.savefig(self.save_path, bbox_inches='tight')



def plot_convergence():
    load_directory = pb.Path('precious_data')
    save_directory = pb.Path('plots')
    save_name = pb.Path('convergence')

    plotter = Plotter(load_directory, save_directory, save_name)

    load_name = [
        '2019_12_10_5_44_2_0_1.npy',
        '2019_12_10_4_8_13_ip_3.npy',
        '2019_12_10_4_12_1_pi_3.npy',
        '2019_12_10_4_9_3_ip_4.npy',
        '2019_12_10_4_21_6_pi_4.npy',
        '2019_12_10_4_35_13_ip_8.npy',
        '2019_12_10_4_35_20_pi_8.npy',
    ]

    goal_thetas = [
        '0',
        r'-$\pi$/3',
        r'$\pi$/3',
        r'-$\pi$/4',
        r'$\pi$/4',
        r'-$\pi$/8',
        r'$\pi$/8',
    ]

    title = 'Policy Convergence'
    xlabel = 'Episode Count'
    ylabel = 'Episode Total Reward'

    fig, ax = plt.subplots()

    plotter.prepare_figure(fig)
    plotter.prepare_axis(ax, title, xlabel, ylabel)
    
    for i in range(len(load_name)):

        fname = load_name[i]
        plotter.load_data(fname)

        cum_rewards = np.array(plotter.get_item('total_reward_arr'))
        ep_rewards = cum_rewards[1:] - cum_rewards[:-1]
        
        ep_count = 5000
        ep_rewards = ep_rewards[:int(ep_count/100)]

        episode_arr = np.linspace(1, ep_count, len(ep_rewards))

        plotter.plot(ax, episode_arr, ep_rewards, label=goal_thetas[i])
        
    plotter.make_legend(ax)
    # plotter.save()
    plt.show()


def plot_torque_weight_study():
    load_directory = pb.Path('rory_data') / pb.Path('control_usage_study')
    save_directory = pb.Path('plots')
    save_name = pb.Path('control_usage')

    plotter = Plotter(load_directory, save_directory, save_name)

    load_name = [
        '2019_12_11_6_31_49_0_1.npy',
        '2019_12_11_5_31_4_0_1.npy',
        '2019_12_11_6_4_16_0_1.npy',
    ]

    torque_weight = [
        '.0001',
        '.00001',
        '.000001',
    ]

    fig, ax = plt.subplots(nrows=2)

    title = 'Control Usage'
    xlabel = 'Time (s)'
    ylabel = 'Applied Torque (Nm)'

    plotter.prepare_figure(fig, 'Comparison of Weight on Applied Torque', height=1.05)
    plotter.prepare_axis(ax[0], title, xlabel, ylabel)
    
    title = 'Pendulum Theta Error'
    xlabel = 'Time (s)'
    ylabel = 'Theta Error (rad)'

    plotter.prepare_axis(ax[1], title, xlabel, ylabel)

    for i in range(len(load_name)):

        fname = load_name[i]
        plotter.load_data(fname)

        torque = np.array(plotter.get_item('torque_arr'))
        el = len(torque)
        torque = torque[:el]

        err = np.array(plotter.get_item('theta_error_arr'))
        err = err[:el]

        t = np.linspace(0, el*.05, len(torque))

        plotter.plot(ax[0], t, torque, label=torque_weight[i])
        plotter.plot(ax[1], t, err, label=torque_weight[i])
        
    plotter.make_legend(ax[0], outside=True)
    plotter.make_legend(ax[1], outside=True)
    # plotter.save()
    plt.show()


def plot_epsilon_study():
    load_directory = pb.Path('rory_data') / pb.Path('epsilon_study')
    save_directory = pb.Path('plots')
    save_name = pb.Path('epsilon_study')

    plotter = Plotter(load_directory, save_directory, save_name)

    load_name = [
        '2019_12_12_2_14_20_0_1.npy',
        '2019_12_12_2_20_58_0_1.npy',
        '2019_12_12_2_24_34_0_1.npy',
    ]

    labels = [
        '.1',
        '.2',
        '.3',
    ]

    fig, ax = plt.subplots(nrows=3)

    title = 'Q Matrix Convergence'
    xlabel = 'Episode Count'
    ylabel = 'Percent Converged'

    plotter.prepare_figure(fig, 'Comparison of Epsilon Values', height=1.05)
    plotter.prepare_axis(ax[0], title, xlabel, ylabel)
    
    title = 'Q Matrix Exploration'
    xlabel = 'Epsiode Count'
    ylabel = 'Percent Explored'

    plotter.prepare_axis(ax[1], title, xlabel, ylabel)

    title = 'Pendulum Theta Error'
    xlabel = 'Time (s)'
    ylabel = 'Theta Error (rad)'

    plotter.prepare_axis(ax[2], title, xlabel, ylabel)

    for i in range(len(load_name)):

        fname = load_name[i]
        plotter.load_data(fname)

        conv_arr = np.array(plotter.get_item('perc_converged_arr'))
        explor_arr = 1 - np.array(plotter.get_item('perc_unexplored_arr'))

        num_episodes = np.array(plotter.get_item('training_episodes'))
        episode_arr = np.linspace(1, num_episodes, len(conv_arr))

        err = np.array(plotter.get_item('theta_error_arr'))

        t = np.linspace(0, len(err)*.05, len(err))

        plotter.plot(ax[0], episode_arr, conv_arr, label=labels[i])
        plotter.plot(ax[1], episode_arr, explor_arr, label=labels[i])
        plotter.plot(ax[2], t, err, label=labels[i])
        
    plotter.make_legend(ax[0], outside=True)
    plotter.make_legend(ax[1], outside=True)
    plotter.make_legend(ax[2], outside=True)
    # plotter.save()
    plt.show()

    
    
if __name__ == '__main__':
    plot_convergence()
    plot_torque_weight_study()
    plot_epsilon_study()

