import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import time


class HeteroLatentAgent(nn.Module):
    def __init__(self, input_shape, args, agent_type):
        super(HeteroLatentAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.latent_out_dim = args.latent_dim
        self.hidden_dim = args.rnn_hidden_dim  # 64
        latent_hidden = args.latent_hidden_dim  # 32

        if agent_type == 0:  # marine
            self.n_agents = args.n_agents - args.n_specialists
            self.n_actions = args.n_actions
        elif agent_type == 1:  # specialists: medivacs or marauders
            self.n_agents = args.n_specialists
            self.n_actions = args.n_special_actions
        else:
            print('agent_type error: ', agent_type)
            raise ValueError("Illegal agent_type!")

        # MI inference
        if args.input_latent:
            self.inference_net = nn.Sequential(nn.Linear(args.latent_dim + input_shape * self.n_agents, latent_hidden))
            self.embed_net = nn.Sequential(nn.Linear(input_shape, latent_hidden))
        else:  # RNN_hidden
            self.inference_net = nn.Sequential(nn.Linear(args.latent_dim + args.rnn_hidden_dim * self.n_agents, latent_hidden))
            self.embed_net = nn.Sequential(nn.Linear(args.rnn_hidden_dim, latent_hidden))
        self.latent_net = nn.Sequential(nn.Linear(args.latent_dim, latent_hidden))

        if self.n_agents == 1:  # BatchNorm1d需要n_agents大于1！
            self.inference_net.add_module(name='1', module=nn.ReLU())
            self.inference_net.add_module(name='2', module=nn.Linear(latent_hidden, args.latent_dim * 2))
            self.embed_net.add_module(name='1', module=nn.ReLU())
            self.embed_net.add_module(name='2', module=nn.Linear(latent_hidden, args.latent_dim * 2))
            self.latent_net.add_module(name='1', module=nn.ReLU())
        else:
            self.inference_net.add_module(name='1', module=nn.BatchNorm1d(latent_hidden))
            self.inference_net.add_module(name='2', module=nn.ReLU())
            self.inference_net.add_module(name='3', module=nn.Linear(latent_hidden, args.latent_dim * 2))
            self.embed_net.add_module(name='1', module=nn.BatchNorm1d(latent_hidden))
            self.embed_net.add_module(name='2', module=nn.ReLU())
            self.embed_net.add_module(name='3', module=nn.Linear(latent_hidden, args.latent_dim * 2))
            self.latent_net.add_module(name='1', module=nn.BatchNorm1d(latent_hidden))
            self.latent_net.add_module(name='2', module=nn.ReLU())

        self.latent = torch.rand(self.n_agents, args.latent_dim * 2)  # (n, mu+var)

        # Decision
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        if args.latent_matmul:
            self.fc2_w_nn = nn.Linear(latent_hidden, args.rnn_hidden_dim * self.n_actions)
            self.fc2_b_nn = nn.Linear(latent_hidden, self.n_actions)
        else:  # Linear FC2 Layer
            self.fc2_latent = nn.Linear(latent_hidden, args.rnn_hidden_dim)
            if not self.args.MI_Disable:  # MI-latent
                self.fc2 = nn.Linear(args.rnn_hidden_dim * 2, self.n_actions)  # [n_agents, hidden+latent]
            else:
                self.fc2 = nn.Linear(args.rnn_hidden_dim, self.n_actions)

    def init_latent(self):
        var_mean = self.latent[:self.n_agents, self.args.latent_dim:].detach().mean()
        return var_mean, self.latent[:self.n_agents, :].detach()

    def forward(self, inputs, hidden_state, t=0, batch_size=0, train_mode=False):
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        x = F.relu(self.fc1(inputs))  # (bs*n,(obs+act+id)) at time t
        h = self.rnn(x, h_in)
        if not self.args.MI_Disable:  # MI-latent
            if self.args.input_latent:  # input->embed->latent
                self.latent = self.embed_net(inputs)
            else:  # decision_latent
                self.latent = self.embed_net(h)
            self.latent[:, -self.latent_out_dim:] = torch.clamp(torch.exp(self.latent[:, -self.latent_out_dim:]), min=self.args.var_floor)

            if train_mode:
                latent_embed = self.latent.reshape(batch_size, self.n_agents, self.latent_out_dim * 2)
                gaussian_embed = Normal(latent_embed[:, :, :self.latent_out_dim],
                                        (latent_embed[:, :, self.latent_out_dim:]) ** (1 / 2))
                latent_sample = gaussian_embed.rsample()
                latent_sample = latent_sample.reshape(batch_size * self.n_agents, self.latent_out_dim)
                # for learner
                latent_embed_ = self.latent.reshape(batch_size, self.n_agents, self.latent_out_dim * 2).mean(axis=1, keepdim=False)
                gaussian_embed_ = Normal(latent_embed_[:, :self.latent_out_dim],
                                         (latent_embed_[:, self.latent_out_dim:]) ** (1 / 2))
                latent_sample_ = gaussian_embed_.rsample()
                latent_sample_ = latent_sample_.reshape(batch_size, self.latent_out_dim)
            else:
                latent_embed = self.latent.reshape(self.n_agents, self.latent_out_dim * 2)  # batch_size=1 ?
                gaussian_embed = Normal(latent_embed[:, :self.latent_out_dim],
                                        (latent_embed[:, self.latent_out_dim:]) ** (1 / 2))
                latent_sample = gaussian_embed.rsample()
            latent = self.latent_net(latent_sample)

            if self.args.latent_matmul:
                h = h.reshape(-1, 1, self.args.rnn_hidden_dim)
                fc2_w = self.fc2_w_nn(latent)
                fc2_b = self.fc2_b_nn(latent)
                fc2_w = fc2_w.reshape(-1, self.args.rnn_hidden_dim, self.n_actions)
                fc2_b = fc2_b.reshape((-1, 1, self.n_actions))
                q = torch.bmm(h, fc2_w) + fc2_b
                h = h.reshape(-1, self.args.rnn_hidden_dim)
            else:  # FC2-Latent FCN Layer
                fc2_latent = self.fc2_latent(latent)
                fc2_input = torch.cat((h, fc2_latent), dim=1)
                q = self.fc2(fc2_input)

            if train_mode:
                return q.view(-1, self.n_actions), h.view(-1, self.args.rnn_hidden_dim), gaussian_embed_, latent_sample_
            else:
                return q.view(-1, self.n_actions), h.view(-1, self.args.rnn_hidden_dim), gaussian_embed, latent_sample
        else:
            q = self.fc2(h)
            return q.view(-1, self.n_actions), h.view(-1, self.args.rnn_hidden_dim), None, None
