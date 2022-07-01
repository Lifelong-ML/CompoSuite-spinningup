#!/usr/bin/env python

import unittest
from functools import partial

import gym

class TestPPO(unittest.TestCase):
    def test_cartpole(self):
        ''' Test training a small agent in a simple environment '''
        env_fn = partial(gym.make, 'CartPole-v1')
        ac_kwargs = dict(hidden_sizes=(32,))
        # TODO: ensure policy has got better at the task


if __name__ == '__main__':
    unittest.main()
