import gym
import numpy as np
import time
import tensorflow as tf
import configparser
import argparse
import os
import sys
from memory_buffer import MemoryBuffer
from feedback import Feedback
from agent import Agent
from teacher import Teacher


def load_config_data(config_dir):
    config = configparser.ConfigParser()
    config.read(config_dir)
    return config


parser = argparse.ArgumentParser()
parser.add_argument('--exp-num', default='-1')
parser.add_argument('--env-name', default='CarRacing-v0', help='CarRacing-v0, SimpleSim-v0')
parser.add_argument('--network-type', default='FNN', help='FNN, RNN')
parser.add_argument('--method', default='1', help='1, 2')
parser.add_argument('--error-prob', default='0', help='1, 2')
args = parser.parse_args()

exp_num = args.exp_num
environment = args.env_name
network = args.network_type
method = args.method
error_prob = args.error_prob

print('\nExperiment number:', exp_num)
print('Environment:', environment)
print('Network:', network)
print('Method:', method, '\n')
time.sleep(3)

# Load common parameters from config file
config = load_config_data('config_files/config.ini')
config_exp_setup = config['EXP_SETUP']

# Load network and method parameters
config = load_config_data('config_files/' + environment + '/' +
                          network + '_m' + method + '.ini')

config_graph = config['GRAPH']
config_buffer = config['BUFFER']
config_general = config['GENERAL']

# Load teacher parameters
config = load_config_data('config_files/' + environment + '/teacher.ini')
config_teacher = config['TEACHER']
config_feedback = config['FEEDBACK']

#network = network
eval_save_folder = '/' + network + '_err20' + method

eval_save_path = config_exp_setup['eval_save_path']
evaluate = config_exp_setup.getboolean('evaluate')
train = config_exp_setup.getboolean('train')
use_teacher = config_exp_setup.getboolean('use_teacher')
show_state = config_exp_setup.getboolean('show_state')
render = config_exp_setup.getboolean('render')
save_results = config_exp_setup.getboolean('save_results')
save_graph = config_exp_setup.getboolean('save_graph')
show_ae_output = config_exp_setup.getboolean('show_ae_output')
show_FPS = config_exp_setup.getboolean('show_FPS')
max_num_of_episodes = config_exp_setup.getint('max_num_of_episodes')
max_time_steps_episode = float(config_exp_setup['max_time_steps_episode'])
history_training_rate = config_buffer.getint('history_training_rate')
use_memory_buffer = config_buffer.getboolean('use')
image_size = config_graph.getint('image_side_length')
resize_observation = config_graph.getboolean('resize_observation')
stop_training = config_exp_setup.getboolean('stop_training')
render_delay = float(config_general['render_delay'])

if not use_memory_buffer:
    eval_save_folder += '_no_buffer'

output_reward_results_name = '/' + network + '_results_' + exp_num + '_'

# Create environment
env = gym.make(environment)

# Create teacher
if use_teacher:
    teacher = Teacher(method=config_teacher['method'],
                      image_size=config_teacher.getint('image_side_length'),
                      dim_a=config_teacher.getint('dim_a'),
                      action_lower_limits=config_teacher['action_lower_limits'],
                      action_upper_limits=config_teacher['action_upper_limits'],
                      loc=config_teacher['loc'],
                      exp=exp_num,
                      error_prob=error_prob,
                      resize_observation=resize_observation)

# Create agent
agent = Agent(train_ae=config_graph.getboolean('train_autoencoder'),
              load_policy=config_graph.getboolean('load'),
              learning_rate=float(config_graph['learning_rate']),
              dim_a=config_graph.getint('dim_a'),
              loss_function_type=config_graph['loss_function_type'],
              policy_loc=config_graph['policy_loc'] + exp_num + '_',
              ae_loc=config_graph['ae_loc'],
              image_size=config_graph.getint('image_side_length'),
              action_upper_limits=config_graph['action_upper_limits'],
              action_lower_limits=config_graph['action_lower_limits'],
              e=config_graph['e'],
              show_ae_output=show_ae_output,
              show_state=show_state,
              resize_observation=resize_observation)

# Create memory buffer
buffer = MemoryBuffer(min_size=config_buffer.getint('min_size'),
                      max_size=config_buffer.getint('max_size'))

# Create feedback object
env.render()
human_feedback = Feedback(env,
                          key_type=config_feedback['key_type'],
                          h_up=config_feedback['h_up'],
                          h_down=config_feedback['h_down'],
                          h_right=config_feedback['h_right'],
                          h_left=config_feedback['h_left'],
                          h_null=config_feedback['h_null'])

# Create saving directory if it does no exist
if save_results:
    if not os.path.exists(eval_save_path + eval_save_folder):
        os.makedirs(eval_save_path + eval_save_folder)

# Initialize variables
init_op = tf.initialize_all_variables()
reward_results = np.array([])
feedback_percentage = np.array([])
total_time = []
last_episode_received_feedback = 1
total_r = 0
t_counter = 0
r = None
h_counter = 0
last_t_counter = 0
init_time = time.time()
stop_training = False
count_down = False

if count_down:
    for i in range(10):
        print(' ' + str(10 - i) + '...')
        time.sleep(1)


# Iterate over the maximum number of episodes
for i_episode in range(max_num_of_episodes):
    print('Starting episode number', i_episode)
    agent.new_episode()
    r = 0
    observation = env.reset()

    if use_teacher:
        teacher.new_episode(i_episode)

    if evaluate and r is not None:
        print('reward episode %i:' % i_episode, r)

    # Iterate over the episode
    for t in range(int(max_time_steps_episode)):
        if render:
            env.render()  # Make the environment visible
            time.sleep(render_delay)  # Add delay to rendering if necessary
                
        # Map action from state
        action = agent.action(observation)

        # Act
        observation, reward, done, info = env.step(action)

        # Accumulate reward
        r += reward

        # Get feedback signal
        if use_teacher:
            h = teacher.get_feedback_signal(observation, action, t_counter)
        else:
            h = human_feedback.get_h()
            # print("Received feedback:", h_counter, "; Total timesteps:", t_counter)

        # Update weights
        if train:
            if np.any(h):  # if any element is not 0
                agent.update(h, observation)
                if not use_teacher:
                    print("feedback", h)
                h_counter += 1
                # Add state action-label pair to memory buffer
                if use_memory_buffer:
                    buffer.add(agent.last_step())

                    # Train sampling from buffer
                    if buffer.initialized():
                        batch = buffer.sample(batch_size=config_buffer.getint('sampling_size'))
                        agent.batch_update(batch)

                        # Train every k time steps
                        if t % history_training_rate == 0:
                            batch = buffer.sample(batch_size=config_buffer.getint('sampling_size'))
                            agent.batch_update(batch)

        t_counter += 1

        # For debugging
        agent.time_step_info(t)

        # Calculate FPS
        if t % 100 == 0 and t != 0 and show_FPS:
            fps = (t_counter - last_t_counter) / (time.time() - init_time)
            init_time = time.time()
            last_t_counter = t_counter
            print('\nFPS:', fps, '\n')

        # End of episode
        if done or human_feedback.ask_for_done():
            reward_results = np.append(reward_results, r)
            if train:
                last_episode_received_feedback = h_counter / (t + 1e6)
            if save_graph:
                agent.save_params()
            if evaluate:
                total_r += r
                print('episode reward:', r)
                print('\n', i_episode, 'avg reward:', total_r / (i_episode + 1), '\n')
                if save_results:
                    np.save(eval_save_path + eval_save_folder + output_reward_results_name + 'reward', reward_results)
                    feedback_percentage = np.append(feedback_percentage, h_counter / (t + 1e-6))
                    np.save(eval_save_path + eval_save_folder + output_reward_results_name + 'feedback', feedback_percentage)
                    total_time.append(t_counter)
                    np.save(eval_save_path + eval_save_folder + output_reward_results_name + 'time', total_time)
                    print('Total time (s):', (time.time() - init_time))
            if use_teacher:
                print('Percentage of given feedback:', (h_counter / (t + 1e-6)) * 100)
            h_counter = 0
            if not use_teacher:
                time.sleep(1)
            break
