from agents.HD_agent import Agent as HD
from agents.LD_agent import Agent as LD


#  Select agent
def agent_selector(network, train_ae, load_policy, learning_rate, dim_a, fc_layers_neurons, loss_function_type,
                   policy_loc, action_upper_limits, action_lower_limits, e, config_graph, config_general):

    if network == 'HD':
        return HD(train_ae=train_ae, load_policy=load_policy, learning_rate=learning_rate, dim_a=dim_a,
                  fc_layers_neurons=fc_layers_neurons, loss_function_type=loss_function_type, policy_loc=policy_loc,
                  action_upper_limits=action_upper_limits, action_lower_limits=action_lower_limits, e=e,
                  ae_loc=config_graph['ae_loc'], image_size=config_graph.getint('image_side_length'),
                  show_ae_output=config_general.getboolean('show_ae_output'),
                  show_state=config_general.getboolean('show_state'),
                  resize_observation=config_general.getboolean('resize_observation'))

    elif network == 'LD':
        return LD(load_policy=load_policy, learning_rate=learning_rate, dim_a=dim_a,
                  fc_layers_neurons=fc_layers_neurons, loss_function_type=loss_function_type, policy_loc=policy_loc,
                  action_upper_limits=action_upper_limits, action_lower_limits=action_lower_limits, e=e,
                  dim_state=config_graph.getint('dim_state'))
    else:
        raise NameError('The selected agent is not valid. Try using: NN or linear_RBFs')