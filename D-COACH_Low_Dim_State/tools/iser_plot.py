import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d
from scipy import stats

#plt.style.use('seaborn')
plt.figure(figsize=(9, 4.5))
plt.rcParams.update({'font.size': 13})

plot_rewards = True
plot_feedbacks = True

plot_type = sys.argv[1]  # time, episode


def get_confidence_interval(data, percentile):
    data = np.sort(data, axis=0)
    X_l, X_r = [], []
    p_l, p_u = [], []
    for i in range(len(data[0])):
        magic_array = []
        for j in range(len(data)):
            magic_array.append(data[j][i])
        magic_array = np.array(magic_array)
        index = np.searchsorted(magic_array, np.mean(magic_array), side='right')
        x_l, x_r = magic_array[:index], magic_array[index:]
        X_l.append(x_l)
        X_r.append(x_r)
        if len(x_l) > 0:
            p_l.append(np.percentile(x_l, 100 - percentile))
        else:
            p_l.append(np.mean(magic_array))

        if len(x_r) > 0:
            p_u.append(np.percentile(x_r, percentile))
        else:
            p_u.append(np.mean(magic_array))
    return p_l, p_u


def get_reward(name, res, type, folder_name):
    reward_results = np.load('../CartPole_OpenAI/results/%s/reward_%s.npy' % (folder_name, name))

    if folder_name == 'DDPG':
        x_raw = np.load('../CartPole_OpenAI/results/%s/timestep_%s.npy' % (folder_name, name)) * (1/22.5)  # to seconda
        x = []
        x.append(x_raw[0])
        for i in range(1, len(x_raw)):
            x.append(x[i-1] + x_raw[i])
        x = np.array(x) / 60  # to minutes
        f = interp1d(np.append(0, x), np.append(0, reward_results))
        x = np.arange(0, x[-1], res)
        reward_results = f(x)

    elif folder_name == 'NN_teacher_Perr' or folder_name == 'rbf_teacher_F2':
        x_raw = np.load('../CartPole_OpenAI/results/%s/time_%s.npy' % (folder_name, name)) * (1 / 22.5)  # to seconds
        x = x_raw
        x = np.array(x) / 60  # to minutes
        f = interp1d(np.append(0, x), np.append(0, reward_results))
        x = np.arange(0, x[-1], res)
        reward_results = f(x)

    elif type == 'time':
        x = np.load('../CartPole_OpenAI/results/%s/time_%s.npy' % (folder_name, name)) / 60.0
        f = interp1d(np.append(0, x), np.append(0, reward_results))
        x = np.arange(0, x[-1], res)
        reward_results = f(x)
    else:
        x = range(len(reward_results))

    return reward_results, x


def plot_reward(reward_results, episodes, name, type=None, var_rewards=None, color=None, alpha=None):

    plt.plot(episodes, reward_results, label=name, color=color, alpha=alpha)

    if var_rewards is not None:
        reward_results_upper = var_rewards[0]
        reward_results_lower = var_rewards[1]
        plt.fill_between(episodes, reward_results_upper, reward_results_lower, color=color, alpha=.1)

    plt.ylabel('Reward')
    if type == 'time':
        plt.xlabel('Time (min)')
    else:
        plt.xlabel('Episode')
    plt.title('Cart-Pole Training Rewards')
    plt.legend(framealpha=0.5)


def mean_plot(experiments, plot_type, folder_name, exp_name, color=None):
    rewards = []
    times = []
    if plot_type == 'time':
        num = 10.2
        res = 0.1
    else:
        num = 32
        res = 1
    training_flags = [num]
    training_flags_index = [-1]
    for i in range(len(experiments)):
        reward_results, time = get_reward(experiments[i], res, plot_type, folder_name)
        print(experiments[i], time)
        rewards.append(reward_results[0:int(num / res)])
        times.append(time[0:int(num / res)])
        # plot_reward(rewards[-1], times[-1], experiments[i], alpha=0.8, type=plot_type)

    plots = np.mean(rewards, axis=0)
    #var_plots = np.var(rewards, axis=0)
    p_l, p_u = get_confidence_interval(rewards, 60)
    var_plots = [p_u, p_l]
    x_plots = times[0]

    plot_reward(plots, x_plots, name=exp_name, type=plot_type, var_rewards=var_plots, color=color)

    axes = plt.gca()
    axes.set_ylim([0, 505])
    if plot_type == 'time':
        axes.set_xlim([0, 10])
    else:
        axes.set_xlim([0, 32])


experiments_humans = ['DiegoAlvarado', 'NicolasMira3', 'LucasNeira3', 'RodrigoPerez3',
                      'NicolasMarticorena3', 'NicolasCruz', 'MatiasMattamala',
                      'IgnacioReyes', 'GabrielAzocar3', 'LerkoAraya']

#experiments = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

experiments = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20']

experiments2 = ['2', '3', '4', '7', '12']

experiments3 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']

experiments4 = ['91', '92', '93', '94', '95', '96', '97', '98', '99', '100',
                '101', '102', '103', '104', '105', '106', '107', '108', '109', '110',
                '111', '112', '113', '114', '115', '116', '117', '118', '119', '120']

mean_plot(experiments4, plot_type, 'NN_teacher_Perr', 'D-COACH simulated teacher', color='C1')
#mean_plot(experiments4, plot_type, 'NN_teacher', 'D-COACH simulated teacher w/ no buffer', color='C2')
mean_plot(experiments3, plot_type, 'rbf_teacher_F2', 'COACH simulated teacher', color='C4')
mean_plot(experiments_humans, plot_type, 'NN_buffer', 'D-COACH human teachers', color='C0')
mean_plot(experiments3, plot_type, 'DDPG', 'DDPG', color='C3')

plt.show()
