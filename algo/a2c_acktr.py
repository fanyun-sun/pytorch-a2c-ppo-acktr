import torch
import torch.nn as nn
import torch.optim as optim

from .kfac import KFACOptimizer

from torch.autograd import Variable as V
import gc

class A2C_ACKTR(object):
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False,
                 pop_art=False):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.pop_art = pop_art

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            self.eps = eps
            self.alpha = alpha
            self.lr = lr
            if self.pop_art:
                self.optimizer = optim.SGD(
                    actor_critic.parameters(), lr)
            else:
                self.optimizer = optim.RMSprop(
                    actor_critic.parameters(), lr, eps=eps, alpha=alpha)
            # self.optimizer = optim.Adam(
                # actor_critic.parameters(), lr=lr, eps=eps)

        if pop_art:
            self.pop_art = True
            device = "cuda"
            self.W     = torch.ones(1, 1).to(device)
            self.b     = torch.zeros(1, 1).to(device)
            self.Sigma = torch.ones(1, 1).to(device)
            self.mu    = torch.zeros(1, 1).to(device)

            self.epsilon = 1e-8
            self.beta  = 0.1

    def reinitialize(self):
            self.optimizer = optim.RMSprop(
                self.actor_critic.parameters(), self.lr, eps=self.eps, alpha=self.alpha)
    
    def update_param(self, Y):
        self.mu_new = self.mu
        self.Sigma_new = self.Sigma
            
        self.mu_new = (1 - self.beta) * self.mu_new + self.beta * Y.mean()
        self.Sigma_new = ((1 - self.beta) * self.Sigma_new ** 2 + self.beta * (Y ** 2).mean()) ** 0.5
        
        self.W = self.W * self.Sigma / self.Sigma_new
        self.b = (self.b * self.Sigma + self.mu - self.mu_new) / self.Sigma
        self.Sigma = self.Sigma_new
        self.mu = self.mu_new

    def output(self, values):
        x = self.W * values + self.b
        x = self.Sigma * x + self.mu
        return x

    def pop_art_update(self, rollouts):
        obs_shape = rollouts.observations.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.states[0].view(-1, self.actor_critic.state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        advantages = rollouts.returns[:-1] - self.output(values)

        # update actor
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.optimizer.zero_grad()
        # (value_loss * self.value_loss_coef + action_loss -
         # dist_entropy * self.entropy_coef).backward()
        (action_loss - dist_entropy * self.entropy_coef).backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        ############
        # update criic here
        ############
        values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.states[0].view(-1, self.actor_critic.state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)

        with torch.no_grad():
            # W, b, Sigma, mu = self.W, self.b, self.Sigma, self.mu
            # update 
            y = rollouts.returns[:-1]
            self.update_param(y) 
           
            # store output
            # prev_output = self.network(state, action) 
            prev_output = values.clone()

            # update network
            scaled_y = (y - self.mu) / (self.Sigma + self.epsilon)
            diff = self.W * prev_output + self.b - scaled_y
            target = (scaled_y - self.b) / (self.W + self.epsilon)

        delta = (self.W ** 2) * (values - target).pow(2)

        avg_delta = delta.mean()
        self.optimizer.zero_grad()
        avg_delta.backward()
        self.optimizer.step()
        
        # update W, b
        self.W -= self.lr * (diff * prev_output).mean()
        self.b -= self.lr * diff.mean()

        return avg_delta.item(), action_loss.item(), dist_entropy.item()

    def update(self, rollouts):
        obs_shape = rollouts.observations.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.states[0].view(-1, self.actor_critic.state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        """
        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False
        """
        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()
