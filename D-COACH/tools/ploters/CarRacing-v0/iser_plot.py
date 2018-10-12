import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d
from scipy import stats

#plt.style.use('seaborn')
plt.rcParams.update({'font.size': 13})
plt.figure(figsize=(9, 4.5))

plot_rewards = True
plot_feedbacks = True

plot_type = sys.argv[1]  # time, episode


def get_exp_names(starter):
    experiments3 = [starter + '1', starter + '2', starter + '3', starter + '4', starter + '5', starter + '6',
                    starter + '7', starter + '8', starter + '9', starter + '10', starter + '11', starter + '12',
                    starter + '13', starter + '14', starter + '15', starter + '16', starter + '17', starter + '18',
                    starter + '19', starter + '20', starter + '21', starter + '22', starter + '23', starter + '24',
                    starter + '25', starter + '26', starter + '27', starter + '28', starter + '29', starter + '30']
    return experiments3


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
    if folder_name == 'ddpg':
        reward_results = np.load('results/CarRacing-v0/%s/reward_results_%s.npy' % (folder_name, name))
    else:
        reward_results = np.load('results/CarRacing-v0/%s/%s_reward.npy' % (folder_name, name))

    if folder_name == 'DDPG' or folder_name == 'FNN_m_test41' or folder_name == 'FNN_m_test5_no_buffer1_no_buffer'\
        or folder_name == 'FNN_err201':
        x_raw = np.load('results/CarRacing-v0/%s/%s_time.npy' % (folder_name, name)) * (1/20.5)  # to seconda
        x = []
        x = x_raw
        x = np.array(x) / 60  # to minutes
        f = interp1d(np.append(0, x), np.append(0, reward_results))
        x = np.arange(0, x[-1], res)
        reward_results = f(x)

    elif folder_name == 'ddpg':
        x_raw = np.load('results/CarRacing-v0/%s/timestep_%s.npy' % (folder_name, name)) * (1/20.5)  # to seconda
        x = []
        x.append(x_raw[0])
        for i in range(1, len(x_raw)):
            x.append(x[i-1] + x_raw[i])
        x = np.array(x) / 60  # to minutes
        f = interp1d(np.append(0, x), np.append(0, reward_results))
        x = np.arange(0, x[-1], res)
        reward_results = f(x)

    elif type == 'time':
        x = np.load('results/CarRacing-v0/%s/%s_time.npy' % (folder_name, name)) / 60.0
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
    plt.title('Car Racing Training Rewards')
    plt.legend(framealpha=0.5)


def get_feedback(network, name):

    feedback_results = np.load('results/CarRacing-v0/' + network + '/' + name[0] + '_feedback.npy')

    #feedback_results = feedback_results[:31]
    episodes = range(feedback_results.shape[0])

    return feedback_results, episodes


def plot_feedback(feedback_results, episodes, network, buffer=False):
    if buffer:
        plt.plot(episodes, 1000*feedback_results, label='% feedback ' + network + '_buffer')
    else:
        plt.plot(episodes, 1000 * feedback_results, label='% feedback ' + network, color='C1')
    plt.legend()


def mean_plot(experiments, plot_type, folder_name, exp_name, color=None):
    rewards = []
    times = []
    if plot_type == 'time':
        num = 20.2
        res = 0.1
    else:
        num = 45
        res = 1
    training_flags = [num]
    training_flags_index = [-1]
    for i in range(len(experiments)):
        reward_results, time = get_reward(experiments[i], res, plot_type, folder_name)
        print(experiments[i], time)
        rewards.append(reward_results[0:int(num / res)])
        times.append(time[0:int(num / res)])
        # plot_reward(rewards[-1], times[-1], experiments[i], alpha=0.8, type=plot_type)

    print('times:', len(times))

    plots = np.mean(rewards, axis=0)
    #var_plots = np.var(rewards, axis=0)
    #var_plots = [np.max(rewards, axis=0), np.min(rewards, axis=0)]
    p_l, p_u = get_confidence_interval(rewards, 60)
    var_plots = [p_u, p_l]
    x_plots = times[0]

    plot_reward(plots, x_plots, name=exp_name, type=plot_type, var_rewards=var_plots, color=color)

    axes = plt.gca()
    axes.set_ylim([-30, 930])
    if plot_type == 'time':
        axes.set_xlim([0, 20])
    else:
        axes.set_xlim([0, 150])


starter = 'FNN_results_'
experiments_humans = [starter + 'DiegoAlvarado', starter + 'NicolasMira', starter + 'LucasNeira3',
                      starter + 'RodrigoPerez3', starter + 'NicolasMarticorena', starter + 'NicolasCruz',
                      starter + 'MatiasMattamala', starter + 'IgnacioReyes', starter + 'GabrielAzocar',
                      starter + 'LerkoAraya']

#alone = [starter + sys.argv[2]]
#alone = [sys.argv[2]]

mean_plot(experiments_humans, plot_type, 'human_iser', 'D-COACH human teachers', color='C0')

experiments3 = get_exp_names(starter)
mean_plot(experiments3, plot_type, 'FNN_m_test41', 'D-COACH simulated teacher', color='C1')
#mean_plot(experiments3, plot_type, 'FNN_m_test5_no_buffer1_no_buffer', 'D-COACH simulated teacher w/ no buffer', color='C2')

experiments3 = get_exp_names('')
mean_plot(experiments3, plot_type, 'ddpg', 'DDPG', color='C3')

#mean_plot(alone, plot_type, 'FNN_err201', 'err20%', color='C0')

# mean_plot(experiments3, plot_type, 'DDPG', 'DDPG', color='C4')

#feedback_results, episodes = get_feedback('FNN_err201', alone)
#plot_feedback(feedback_results, episodes, 'FNN_err201')

plt.show()
