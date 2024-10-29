import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HeteroQMixer(nn.Module):
    def __init__(self, args, mixer_type):
        super(HeteroQMixer, self).__init__()
        self.args = args
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim

        if mixer_type == 0:  # marines
            self.n_agents = (args.n_agents - args.n_medivacs)
        elif mixer_type == 1:  # medivac
            self.n_agents = args.n_medivacs
        elif mixer_type == -1:  # universal
            self.n_agents = (args.n_agents - args.n_medivacs)
            self.n_medivacs = args.n_medivacs
            if args.hier_qmix:
                self.n_agent_type = args.n_agent_type
                if args.vs_input:
                    self.n_agent_type += 1
        else:
            print('mixer_type error: ', mixer_type)
            raise ValueError("Illegal mixer_type!")

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(),
                                               nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        if args.universal_qmix:
            if getattr(args, "hypernet_layers", 1) == 1:
                self.hyper_w_1_med = nn.Linear(self.state_dim, self.embed_dim * self.n_medivacs)
                self.hyper_w_final_med = nn.Linear(self.state_dim, self.embed_dim)
            if getattr(args, "hypernet_layers", 1) == 2:
                hypernet_embed = self.args.hypernet_embed
                self.hyper_w_1_med = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                                   nn.ReLU(),
                                                   nn.Linear(hypernet_embed, self.embed_dim * self.n_medivacs))
                self.hyper_w_final_med = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                                       nn.ReLU(),
                                                       nn.Linear(hypernet_embed, self.embed_dim))
            self.hyper_b_1_med = nn.Linear(self.state_dim, self.embed_dim)
            if args.hier_qmix:
                hypernet_embed = self.args.hypernet_embed
                self.hyper_w_1_hier = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                                    nn.ReLU(),
                                                    nn.Linear(hypernet_embed, self.embed_dim * self.n_agent_type))
                self.hyper_w_final_hier = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                                        nn.ReLU(),
                                                        nn.Linear(hypernet_embed, self.embed_dim))
                self.hyper_b_1_hier = nn.Linear(self.state_dim, self.embed_dim)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, medivac_qs=None):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        if self.args.relu_qmix:
            w1 = F.relu(self.hyper_w_1(states))
            w_final = F.relu(self.hyper_w_final(states))
        else:
            w1 = th.abs(self.hyper_w_1(states))
            w_final = th.abs(self.hyper_w_final(states))
        # First layer
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)

        # Compute final output
        if self.args.universal_qmix:
            # Heterogeneous Agents
            assert medivac_qs is not None
            medivac_qs = medivac_qs.view(-1, 1, self.n_medivacs)
            if self.args.relu_qmix:
                w1_m = F.relu(self.hyper_w_1_med(states))
                w_final_m = F.relu(self.hyper_w_final_med(states))
            else:
                w1_m = th.abs(self.hyper_w_1_med(states))
                w_final_m = th.abs(self.hyper_w_final_med(states))
            b1_m = self.hyper_b_1_med(states)
            w1_m = w1_m.view(-1, self.n_medivacs, self.embed_dim)
            b1_m = b1_m.view(-1, 1, self.embed_dim)
            hidden_m = F.elu(th.bmm(medivac_qs, w1_m) + b1_m)
            w_final_m = w_final_m.view(-1, self.embed_dim, 1)
            if self.args.hier_qmix:
                q_agent = th.bmm(hidden, w_final)  # (-1, 1, 1)
                q_med = th.bmm(hidden_m, w_final_m)
                if self.args.vs_input:
                    q_input = th.cat((q_agent, q_med, v), dim=1)
                else:
                    q_input = th.cat((q_agent, q_med), dim=1)
                q_input = q_input.view(-1, 1, self.n_agent_type)
                # q_input = q_input.view(-1, 1, (self.n_agent_type+1))
                if self.args.relu_qmix:
                    w1_h = F.relu(self.hyper_w_1_hier(states))
                    w_final_h = F.relu(self.hyper_w_final_hier(states))
                else:
                    w1_h = th.abs(self.hyper_w_1_hier(states))
                    w_final_h = th.abs(self.hyper_w_final_hier(states))
                b1_h = self.hyper_b_1_hier(states)
                # w1_h = w1_h.view(-1, (self.n_agent_type+1), self.embed_dim)
                w1_h = w1_h.view(-1, self.n_agent_type, self.embed_dim)
                b1_h = b1_h.view(-1, 1, self.embed_dim)
                hidden_h = F.elu(th.bmm(q_input, w1_h) + b1_h)
                w_final_h = w_final_h.view(-1, self.embed_dim, 1)
                y = th.bmm(hidden_h, w_final_h) + v
            else:
                y = th.bmm(hidden, w_final) + th.bmm(hidden_m, w_final_m) + v
        else:
            y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
