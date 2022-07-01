CompoSuite-Spinning Up
======================
This repository is a modified version of the original [Spinning Up](https://github.com/openai/spinningup) repository by Joshua Achiam, adapted to work as the base training code for [CompoSuite](https://github.com/Lifelong-ML/CompoSuite). The main modifications from the original repository are:

1. **Removal of TensorFlow code.** The version restriction `tensorflow<2.0` caused conflicts with `robosuite`, and therefore all TensorFlow dependencies were removed. Since CompoSuite was evaluated using PyTorch, TensorFlow is not necessary. 
2. **Addition of compositional and multi-task PPO.** The main additions of this version of Spinning Up are mechanisms to train PPO over multiple tasks either with a monolithic architecture or with a modular architecture. The training algorithm assumes that each task runs on a separate MPI process.
3. **Tanh policy activation.** We applied a tanh activation to the output of the Gaussian policy for PPO training, to prevent the agent from outputting overconfident actions.
4. **Fixed variance Gaussian policy.** We found empirically that CompoSuite tasks became much more easily learnable if the variance of the Gaussian policy used to train PPO was fixed instead of learned, and therefore used this formulation.

---
**Below is the README for the original Spinning Up repository.**


**Status:** Maintenance (expect bug fixes and minor updates)

Welcome to Spinning Up in Deep RL! 
==================================

This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with [deep learning](http://ufldl.stanford.edu/tutorial/).

This module contains a variety of helpful resources, including:

- a short [introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to RL terminology, kinds of algorithms, and basic theory,
- an [essay](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) about how to grow into an RL research role,
- a [curated list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) of important papers organized by topic,
- a well-documented [code repo](https://github.com/openai/spinningup) of short, standalone implementations of key algorithms,
- and a few [exercises](https://spinningup.openai.com/en/latest/spinningup/exercises.html) to serve as warm-ups.

Get started at [spinningup.openai.com](https://spinningup.openai.com)!


Citing Spinning Up
------------------

If you reference or use Spinning Up in your research, please cite:

```
@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}
```