import numpy as np
import matplotlib.pyplot as plt
import sys

plot_rewards = True
plot_feedbacks = True

compare_NN = False
compare_buffer = False

network = 'RNN_m1'
buffer = True
total_n = 30

select_n = True
n = sys.argv[1]


def get_reward(network, buffer, number):
    if buffer:
        reward_results = np.load('results/' + network + '/' + network[:-3] + '_results_' + str(number) + '_reward.npy')
    else:
        if network is not 'DDPG':
            reward_results = np.load('results/' + network + '_sec/' + network[:-3] + '_results_' + str(number) + '_reward.npy')
        else:
            reward_results = np.load('results/' + network + '/reward_results_' + str(number) + '.npy')

    episodes = range(reward_results.shape[0])

    return reward_results, episodes


def get_feedback(network, buffer, number):
    if buffer:
        feedback_results = np.load('results/' + network + '/' + network[:-3] + '_results_' + str(number) + '_feedback.npy')
    else:
        feedback_results = np.load('results/' + network + '_sec/' + network + '_results_' + str(number) + '_feedback.npy')

    #feedback_results = feedback_results[:31]
    episodes = range(feedback_results.shape[0])

    return feedback_results, episodes


def plot_reward(reward_results, episodes, network, buffer, var_rewards=None, color=None):
    if buffer:
        labels = {'NN':'D-COACH', 'DDPG':'DDPG', 'human':'Human'}
    else:
        labels = {'NN':'D-COACH no buffer', 'DDPG':'DDPG', 'human':'Human'}

    plt.plot(episodes, reward_results, label='D-COACH', color=color)

    if var_rewards is not None:
        std_dev = np.sqrt(var_rewards)
        reward_results_upper = np.clip(reward_results + std_dev, 0, 930)
        reward_results_lower = reward_results - std_dev
        plt.fill_between(episodes, reward_results_upper, reward_results_lower, color=color, alpha=.1)
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.title('Racing Car Training Rewards')
    plt.legend()


def plot_feedback(feedback_results, episodes, network, buffer):
    if buffer:
        plt.plot(episodes, 400*feedback_results, label='% feedback ' + network + '_buffer')
    else:
        plt.plot(episodes, 400 * feedback_results, label='% feedback ' + network)
    plt.legend()


if plot_rewards:
    if select_n:
        reward_results, episodes = get_reward(network, buffer, n)
        plot_reward(reward_results, episodes, network, buffer)
    elif compare_NN:
        total_rewards = []
        for i in range(total_n):
            reward_results, episodes = get_reward('NN', buffer, i+1)
            total_rewards.append(reward_results)
        mean_rewards = np.mean(total_rewards, axis=0)
        plot_reward(mean_rewards, episodes, 'NN', buffer)

        total_rewards = []
        for i in range(total_n):
            reward_results, episodes = get_reward('NN_lstm', buffer, i+1)
            total_rewards.append(reward_results)
        mean_rewards = np.mean(total_rewards, axis=0)
        plot_reward(mean_rewards, episodes, 'NN_lstm', buffer)
    elif compare_buffer:
        reward_results, episodes = get_reward('human', False, 1)
        plot_reward(reward_results[:150], range(reward_results[:150].shape[0]), 'human', False, color='C4')

        total_rewards = []
        for i in range(total_n):
            reward_results, episodes = get_reward(network, True, i+1)
            total_rewards.append(reward_results)
        mean_rewards = np.mean(total_rewards, axis=0)
        var_rewards = np.var(total_rewards, axis=0)
        plot_reward(mean_rewards, episodes, network, True, var_rewards=var_rewards, color='C0')

        total_rewards = []
        for i in range(total_n):
            reward_results, episodes = get_reward(network, False, i+1)
            total_rewards.append(reward_results)
        mean_rewards = np.mean(total_rewards, axis=0)
        var_rewards = np.var(total_rewards, axis=0)
        plot_reward(mean_rewards, episodes, network, False, var_rewards=var_rewards, color='C1')

        total_rewards = []
        for i in range(total_n):
            reward_results, episodes = get_reward('DDPG', False, i+1)
            total_rewards.append(reward_results)
        mean_rewards = np.mean(total_rewards, axis=0)
        var_rewards = np.var(total_rewards, axis=0)
        plot_reward(mean_rewards, episodes, 'DDPG', False, var_rewards=var_rewards, color='C3')
    else:
        total_rewards = []
        for i in range(total_n):
            reward_results, episodes = get_reward(network, buffer, i+1)
            total_rewards.append(reward_results)
        mean_rewards = np.mean(total_rewards, axis=0)
        var_rewards = np.var(total_rewards, axis=0)
        plot_reward(mean_rewards, episodes, network, buffer, var_rewards=var_rewards)

if plot_feedbacks and not compare_NN and not compare_buffer:
    if select_n:
        feedback_results, episodes = get_feedback(network, buffer, n)
        plot_feedback(feedback_results, episodes, network, buffer)
    else:
        total_feedbacks = []
        for i in range(total_n):
            feedback_results, episodes = get_feedback(network, buffer, i+1)
            total_feedbacks.append(feedback_results)
        mean_feedbacks = np.mean(total_feedbacks, axis=0)
        plot_feedback(mean_feedbacks, episodes, network, buffer)

#axes = plt.gca()
#axes.set_ylim([0, 940])
#axes.set_xlim([0, 149])
plt.show()
