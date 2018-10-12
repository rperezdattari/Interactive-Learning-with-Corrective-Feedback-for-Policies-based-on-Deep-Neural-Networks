from simulated_teacher.HD_teacher import Teacher as HD
from simulated_teacher.LD_teacher import Teacher as LD


#  Select teacher
def teacher_selector(network, dim_a, action_lower_limits, action_upper_limits, loc, error_prob, teacher_parameters,
                     config_general, config_teacher):

    if network == 'HD':
        return HD(dim_a=dim_a, action_lower_limits=action_lower_limits, action_upper_limits=action_upper_limits,
                  loc=loc, error_prob=error_prob, teacher_parameters=teacher_parameters,
                  image_size=config_teacher.getint('image_side_length'),
                  resize_observation=config_general.getboolean('resize_observation'))

    elif network == 'LD':
        return LD(dim_a=dim_a, action_lower_limits=action_lower_limits, action_upper_limits=action_upper_limits,
                  loc=loc, error_prob=error_prob, teacher_parameters=teacher_parameters,
                  dim_state=config_teacher.getint('dim_state'))
    else:
        raise NameError('Not valid network.')
