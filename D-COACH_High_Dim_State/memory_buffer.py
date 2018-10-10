import random

"""Base code of this file extracted from: https://github.com/fhennecker/deepdoom/blob/master/src/memory.py"""


class MemoryBuffer:
    def __init__(self, min_size=20, max_size=1000):
        self.feedback_steps = []
        self.min_size, self.max_size = min_size, max_size

    def full(self):
        return len(self.feedback_steps) >= self.max_size

    def initialized(self):
        return len(self.feedback_steps) >= self.min_size

    def add(self, feedback_step):
        if self.full:
            self.feedback_steps.pop(0)
        self.feedback_steps.append(feedback_step)

    def sample(self, batch_size):
        return [random.choice(self.feedback_steps) for i in range(batch_size)]
