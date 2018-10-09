import gym
from COACH import COACH
from feedback import Feedback
import time
import numpy as np
import sys
import os

results_path = './results'
exp_name = str(sys.argv[1])
exp = str(sys.argv[2])
aprox_function = str(sys.argv[3])
buffer = bool(sys.argv[4])
# Create environment
env = gym.make('CartPole-v0')
env.render()
feedback = Feedback(env)
agent = COACH(aprox_function, buffer)
teacher = COACH('Teacher')
maxNumOfEpisodes = 50000
maxTimeStepsEpisode = 1000
reward = 0
reward_results = []
received_feedback = []
total_time = []
FPS_mean = []
output_reward_results_name = 'reward_results_DCOACH_cartpole2'
# Iterate over the maximum number of episodes
time.sleep(1.0)
use_teacher = True
train = True
render = False
save_results = True
save_params = True
count_down = False
show_FPS = False
last_episode_received_feedback = 1
histoy_training_rate = 10
t_counter = 0
last_t_counter = 0
init_time = time.time()

if count_down:
    for i in range(10):
        print(' ' + str(10 - i) + '...')
        time.sleep(1)

# Create saving directory if it does no exist
if save_results:
    if not os.path.exists(results_path + exp):
        os.makedirs(results_path + exp)

start_time = time.time()
total_timestep = 0
for i_episode in range(maxNumOfEpisodes):
    observation = env.reset()  # If the environment is reset, the first observation is given
    agent.new_episode()  # Reset episode variables
    print('Starting episode number', i_episode)
    # Iterate over all episodes

    if aprox_function == 'NN':
        agent.buffer_auto_training(last_episode_received_feedback)
    h_counter = 0
    for t in range(maxTimeStepsEpisode):
        if render:
            env.render()  # Make the environment visible
        action = agent.action(observation)
        observation, r, done, info = env.step(action)  # Receive an observation of the environment and reward after action
        h = feedback.get_h()
        reward += r
        if use_teacher:
            h = teacher.get_feedback_signal(observation, action, total_timestep)
        if h != 0 and train:
            agent.update(h, observation)
            h_counter += 1
            if render:
                if h == 1:
                    print('h: RIGHT')
                else:
                    print('h: LEFT')

        if t % histoy_training_rate == 0 and buffer and aprox_function == 'NN' and train:
            agent.train_NN_model_from_buffer()

        t_counter += 1

        # Calculate FPS
        if t_counter % 100 == 0 and t != 0 and show_FPS:
            fps = (t_counter - last_t_counter) / (time.time() - init_time)
            FPS_mean.append(fps)
            init_time = time.time()
            last_t_counter = t_counter
            print('\nFPS mean:', np.mean(FPS_mean), '\n')

        if render:
            time.sleep(0.035)

        total_timestep += 1

        if done:  # If the episode is finished
            if train:
                last_episode_received_feedback = h_counter / (t + 1e-6)
            if save_params:
                agent.save_params(exp_name)
            print('episode reward:', reward)
            if save_results:
                reward_results.append(reward)
                np.save(results_path + exp + '/reward_' + exp_name, reward_results)
                received_feedback.append(h_counter / t)
                np.save(results_path + exp + '/feedback_' + exp_name, received_feedback)
                total_time.append(t_counter)
                np.save(results_path + exp + '/time_' + exp_name, total_time)
                #print('reward results:', reward_results)
                print('Feedback received:', h_counter / t)
                print('Total time (s):', (time.time() - start_time))
                if (t_counter/(22.5*60)) > 11:
                    env.env.close()
                    exit()

                #if np.mean(reward_results[-5:]) == 500:
                 #   train = False
                  #  print('Stopped training.')
            reward = 0
            if render:
                time.sleep(0.5)
            break
