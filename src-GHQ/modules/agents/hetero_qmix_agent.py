import torch.nn as nn
import torch.nn.functional as F


class HeteroQmixAgent(nn.Module):
    def __init__(self, input_shape, args, agent_type):
        super(HeteroQmixAgent, self).__init__()
        self.args = args
        self.agent_type = agent_type
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # if agent_type == -1:  # marine+medivac——实际上并不行，因为input_shape不同
        #     self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        #     self.fc_med = nn.Linear(args.rnn_hidden_dim, args.n_special_actions)
        if agent_type == 0:  # marine
            self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        elif agent_type == 1:  # medivac
            self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_special_actions)
        else:
            print('agent_type error: ', agent_type)
            raise ValueError("Illegal agent_type!")

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        # if self.agent_type == -1:
        #     q_med = self.fc_med(h)
        #     return q, q_med, h
        # else:
        return q, h
