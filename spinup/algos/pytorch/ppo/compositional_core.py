import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from spinup.algos.pytorch.ppo.core import Actor
import copy

# module_inputs = {'obs_state': np.array([0, 1,2, 3,4]), 'robot_state': np.array([5,6,7])}
# obs = np.array() # 1, 70
# module_assignment_positions = {'obstacle_module_onehot': np.array([65, 66, 67, 68])}

class compositional_mlp(nn.Module):
    def __init__(
        self,
        sizes,
        activation,
        num_modules, 
        module_assignment_positions, 
        module_inputs, 
        interface_depths,
        graph_structure,
        output_activation=nn.Identity
    ):
        super().__init__()
        self._num_modules = num_modules
        self.module_assignment_positions = module_assignment_positions
        self._module_inputs = module_inputs         # keys in a dict
        self._interface_depths = interface_depths
        self._graph_structure = graph_structure     # [[0], [1,2], 3] or [[0], [1], [2], [3]]   

        self._module_list = nn.ModuleList() # e.g., object, robot, task...
        
        for graph_depth in range(len(graph_structure)): # root -> children -> ... leaves 
            for j in graph_structure[graph_depth]:          # loop over all module types at this depth
                self._module_list.append(nn.ModuleDict())   # pre, post
                self._module_list[j]['pre_interface'] = nn.ModuleList()
                self._module_list[j]['post_interface'] = nn.ModuleList()
                
                for k in range(num_modules[j]):                 # loop over all modules of this type
                    layers_pre = []
                    layers_post = []
                    for i in range(len(sizes[j]) - 1):              # loop over all depths in this module
                        act = activation if graph_depth < len(graph_structure) - 1 or i < len(sizes[j])-2 else output_activation

                        if i == interface_depths[j]:
                            input_size = sum(sizes[j_prev][-1] for j_prev in graph_structure[graph_depth - 1])
                            input_size += sizes[j][i]
                        else:
                            input_size = sizes[j][i]

                        new_layer = [nn.Linear(input_size, sizes[j][i+1]), act()]
                        if i < interface_depths[j]:
                            layers_pre += new_layer
                        else:
                            layers_post += new_layer
                    if layers_pre:
                        self._module_list[j]['pre_interface'].append(nn.Sequential(*layers_pre))
                    else:   # it's either a root or a module with no preprocessing
                        self._module_list[j]['pre_interface'].append(nn.Identity())
                    self._module_list[j]['post_interface'].append(nn.Sequential(*layers_post))

    def forward(self, input_val):
        x = None
        for graph_depth in range(len(self._graph_structure)):     # root -> children -> ... -> leaves
            x_post = []
            for j in self._graph_structure[graph_depth]:          # nodes (modules) at this depth
                if len(input_val.shape) == 1:
                    x_pre = input_val[self._module_inputs[j]]
                    onehot = input_val[self.module_assignment_positions[j]]
                else:
                    x_pre = input_val[:, self._module_inputs[j]]
                    onehot = input_val[0, self.module_assignment_positions[j]]
                    assert (input_val[:, self.module_assignment_positions[j]] == onehot).all()
                module_index = onehot.argmax()

                x_pre = self._module_list[j]['pre_interface'][module_index](x_pre)
                if x is not None: x_pre = torch.cat((x, x_pre), dim=-1)
                x_post.append(self._module_list[j]['post_interface'][module_index](x_pre))
            x = torch.cat(x_post, dim=-1)
        return x

class CompositionalMLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, 
        hidden_sizes,
        module_assignment_positions,
        module_inputs,
        interface_depths,
        graph_structure,
        activation):

        super().__init__()

        sizes = list(hidden_sizes)
        for j in range(len(sizes)):
            input_size = len(module_inputs[j])
            sizes[j] = [input_size] + list(sizes[j])
            if j in graph_structure[-1]:
                sizes[j] = sizes[j] + [act_dim]

        self.logits_net = compositional_mlp(sizes=sizes,
            activation=activation,
            num_modules=num_modules,
            module_assignment_positions=module_assignment_positions,
            module_inputs=module_inputs,
            interface_depths=interface_depths,
            graph_structure=graph_structure)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class CompositionalMLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, 
        hidden_sizes,
        num_modules,
        module_assignment_positions,
        module_inputs,
        interface_depths,
        graph_structure,
        activation,
        log_std_init):

        super().__init__()

        sizes = list(hidden_sizes)
        for j in range(len(sizes)):
            input_size = len(module_inputs[j])
            sizes[j] = [input_size] + list(sizes[j])
            if j in graph_structure[-1]:
                sizes[j] = sizes[j] + [act_dim]

        log_std = log_std_init * np.ones(act_dim, dtype=np.float32)
        log_std[-1] = -0.5
        self.log_std = torch.as_tensor(log_std)

        self.mu_net = compositional_mlp(sizes=sizes,
            activation=activation, 
            output_activation=nn.Tanh,
            num_modules=num_modules,
            module_assignment_positions=module_assignment_positions,
            module_inputs=module_inputs,
            interface_depths=interface_depths,
            graph_structure=graph_structure)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        # Last axis sum needed for Torch Normal distribution
        return pi.log_prob(act).sum(axis=-1)

class CompositionalMLPCritic(nn.Module):

    def __init__(self, obs_dim, 
        hidden_sizes,
        num_modules,
        module_assignment_positions,
        module_inputs,
        interface_depths,
        graph_structure,
        activation):

        super().__init__()

        sizes = list(hidden_sizes)
        for j in range(len(sizes)):
            input_size = len(module_inputs[j])
            sizes[j] = [input_size] + list(sizes[j])
            if j in graph_structure[-1]:
                sizes[j] = sizes[j] + [1]

        self.v_net = compositional_mlp(sizes=sizes,
            activation=activation,
            num_modules=num_modules,
            module_assignment_positions=module_assignment_positions,
            module_inputs=module_inputs,
            interface_depths=interface_depths,
            graph_structure=graph_structure)

    def forward(self, obs):
        # Critical to ensure v has right shape.
        return torch.squeeze(self.v_net(obs), -1)

class CompositionalMLPActorCritic(nn.Module):
    def __init__(self, obs_dim, action_space, observation_positions,
                 hidden_sizes,
                 module_names,
                 module_input_names,
                 interface_depths,
                 graph_structure, 
                 activation=nn.Tanh,
                 log_std_init=-0.5):
        super().__init__()
        # policy builder depends on action space
        self.module_assignment_positions = [observation_positions[key] for key in module_names]
        self.num_modules = [len(onehot_pos) for onehot_pos in self.module_assignment_positions]
        self.module_inputs = [observation_positions[key] for key in module_input_names]
        self.graph_structure = graph_structure
        if isinstance(action_space, Box):
            self.pi = CompositionalMLPGaussianActor(obs_dim, action_space.shape[0], 
                hidden_sizes=hidden_sizes,
                num_modules=self.num_modules,
                module_assignment_positions=self.module_assignment_positions,
                module_inputs=self.module_inputs,
                interface_depths=interface_depths,
                graph_structure=graph_structure,
                activation=activation,
                log_std_init=log_std_init)
        elif isinstance(action_space, Discrete):
            self.pi = CompositionalMLPCategoricalActor(obs_dim, action_space.n, 
                hidden_sizes=hidden_sizes,
                num_modules=self.num_modules,
                module_assignment_positions=self.module_assignment_positions,
                module_inputs=self.module_inputs,
                interface_depths=interface_depths,
                graph_structure=graph_structure,
                activation=activation)

        # build value function
        self.v  = CompositionalMLPCritic(obs_dim, 
            hidden_sizes=hidden_sizes,
            num_modules=self.num_modules,
            module_assignment_positions=self.module_assignment_positions,
            module_inputs=self.module_inputs,
            interface_depths=interface_depths,
            graph_structure=graph_structure, 
            activation=activation)

    def step(self, obs, deterministic=False):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs, deterministic=False):
        return self.step(obs, deterministic=deterministic)[0]

