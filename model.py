import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import init, init_normc_


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self,
                 obs_shape,
                 action_space,
                 recurrent_policy,
                 hidden_size,
                 args):

        super(Policy, self).__init__()
        if len(obs_shape) == 3:
            self.base = CNNBase(obs_shape[0], recurrent_policy)
        elif len(obs_shape) == 1:
            assert not recurrent_policy, \
                "Recurrent policy is not implemented for the MLP controller"
            self.base = MLPBase(obs_shape[0], hidden_size, args)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.state_size = self.base.state_size
        self.leaky = args.leaky
        self.scale = 1.

    def rescale(self, ratio):
        for idx, m in enumerate(self.base.critic1.modules()):
            if isinstance(m, nn.Linear):
                m.weight.data.mul_(ratio ** (1./3))
                m.bias.data.mul_(ratio ** ((idx+1)/3))

        self.scale *= ratio
        self.base.scale = self.scale

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks):
        value, _, _ = self.base(inputs, states, masks)
        value /= self.scale
        return value
    
    def base_forward(self, inputs):
        ret = []

        x = inputs
        for m in self.base.critic1.modules():
            if isinstance(m, nn.Sequential):
                continue
            x = m(x)
            if self.leaky:
                if isinstance(m, nn.LeakyReLU):
                    ret.append(x.clone())
            else:
                if isinstance(m, nn.ReLU):
                    ret.append(x.clone())

        assert len(ret) == 2
        return ret


    def evaluate_actions(self, inputs, states, masks, action):
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, states


class CNNBase(nn.Module):
    def __init__(self, num_inputs, use_gru):
        super(CNNBase, self).__init__()

        init_ = lambda m: init(m,
                      nn.init.orthogonal_,
                      lambda x: nn.init.constant_(x, 0),
                      nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, 512)),
            nn.ReLU()
        )

        if use_gru:
            self.gru = nn.GRUCell(512, 512)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        init_ = lambda m: init(m,
          nn.init.orthogonal_,
          lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(512, 1))

        self.train()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    @property
    def output_size(self):
        return 512

    def forward(self, inputs, states, masks):
        x = self.main(inputs / 255.0)

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
                N = states.size(0)
                T = int(x.size(0) / N)

                # unflatten
                x = x.view(T, N, x.size(1))

                # Same deal with masks
                masks = masks.view(T, N, 1)

                outputs = []
                for i in range(T):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)

                # assert len(outputs) == T
                # x is a (T, N, -1) tensor
                x = torch.stack(outputs, dim=0)
                # flatten
                x = x.view(T * N, -1)

        return self.critic_linear(x), x, states

class MLPBase(nn.Module):
    def __init__(self, num_inputs, hidden_size, args):
        super(MLPBase, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, 64)),
            nn.ReLU(),
            init_(nn.Linear(64, 64)),
            nn.Tanh()
        )


        # self.linear1 = init_(nn.Linear(num_inputs, 64))
        # self.linear2 = init_(nn.Linear(64, 64))
        # self.linear3 = init_(nn.Linear(64, 1))
        if args.leaky:
            self.critic1 = nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)),
                nn.LeakyReLU(),
                init_(nn.Linear(hidden_size, hidden_size)),
                nn.LeakyReLU(),
                init_(nn.Linear(hidden_size,1)),
            )

            self.critic2 = nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)),
                nn.LeakyReLU(),
                init_(nn.Linear(hidden_size, hidden_size)),
                nn.LeakyReLU(),
                init_(nn.Linear(hidden_size,1)),
            )
        elif args.elu:
            self.critic1 = nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)),
                nn.ELU(),
                init_(nn.Linear(hidden_size, hidden_size)),
                nn.ELU(),
                init_(nn.Linear(hidden_size,1)),
            )

            self.critic2 = nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)),
                nn.ELU(),
                init_(nn.Linear(hidden_size, hidden_size)),
                nn.ELU(),
                init_(nn.Linear(hidden_size,1)),
            )
            
        else:
            self.critic1 = nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)),
                nn.ReLU(),
                init_(nn.Linear(hidden_size, hidden_size)),
                nn.ReLU(),
                init_(nn.Linear(hidden_size,1)),
            )

            self.critic2 = nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)),
                nn.ReLU(),
                init_(nn.Linear(hidden_size, hidden_size)),
                nn.ReLU(),
                init_(nn.Linear(hidden_size,1)),
            )

        # self.critic_linear = init_(nn.Linear(64, 1))
        self.train()
        self.scale = 1.
        self.network_ratio = args.network_ratio

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 64

    def forward(self, inputs, states, masks):

        # self.relu1 = F.relu(self.linear1(inputs))
        # self.relu2 = F.relu(self.linear2(self.relu1))
        # self.critic = self.linear3(self.relu2)

        # hidden_critic = (self.critic1(inputs)/self.scale + self.critic2(inputs))/2
        # hidden_critic = self.network_ratio * self.critic1(inputs)/self.scale + (1-self.network_ratio) * self.critic2(inputs)
        hidden_critic = self.critic1(inputs)
        # hidden_critic = (self.critic1(inputs) + self.critic2(inputs))/2
        hidden_actor = self.actor(inputs)

        # return self.critic_linear(hidden_critic), hidden_actor, states
        return hidden_critic, hidden_actor, states

if __name__=='__main__':
    model = MLPBase(1)
    x = torch.FloatTensor([[1], [-1]])
    for m in model.critic1.modules():
        x = m(x)
        print('after', m, x)

    for idx, m in enumerate(model.critic1.modules()):
        if isinstance(m, nn.Linear):
            print(idx, m)
        if isinstance(m, nn.ReLU):
            print(idx, m)
