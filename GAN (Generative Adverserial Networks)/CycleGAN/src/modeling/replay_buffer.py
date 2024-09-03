import torch

import random

from torch.autograd import Variable


# we utilize the idea of using a set of generated images to test the discriminator
# insted of just passing the latest generated image
# this is a strategy to stabilitze training CycleGAN
class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.data = []  # the buffer

    def push_and_pop(self, data):
        to_be_returned = []

        for element in data.data:
            element = torch.unsqueeze(element, 0)  # adds an extra dimension

            if len(self.data) < self.max_size:  # there is some space in the buffer
                self.data.append(element)
                to_be_returned.append(element)
            else:
                # returns a newly added image with a probability of 0.5
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)

                    to_be_returned.append(self.data[i].clone())
                    self.data[i] = element
                else:  # else, return an older image
                    to_be_returned.append(element)

        return Variable(torch.cat(to_be_returned))
