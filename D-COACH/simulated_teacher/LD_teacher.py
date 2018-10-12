import numpy as np
from simulated_teacher.teacher_base import TeacherBase


class Teacher(TeacherBase):
    def __init__(self, dim_a=3, action_lower_limits='0,0,0', action_upper_limits='1,1,1',
                 loc='graphs/teacher/CarRacing-v0/network', error_prob=0, teacher_parameters='0.6,0.00001',
                 dim_state=4):

        super(Teacher, self).__init__(dim_a=dim_a, action_lower_limits=action_lower_limits,
                                      action_upper_limits=action_upper_limits, loc=loc,
                                      error_prob=error_prob, teacher_parameters=teacher_parameters)

        self.low_dim_input_shape = dim_state

    def _preprocess_observation(self, observation):
        self.low_dim_observation = np.reshape(observation, [-1, self.low_dim_input_shape])

