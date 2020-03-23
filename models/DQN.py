import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, dims, n_actions, conv_net):
        super(DQN, self).__init__()
        self.conv_net = conv_net
        self.n_actions = n_actions
        self.layers = nn.ModuleList()
        prev_dim = dims[0]
        for dim in dims[1:]:
            self.layers.append(nn.Linear(prev_dim, dim))
            self.layers.append(nn.ReLU())
            prev_dim = dim
        self.layers.append(nn.Linear(prev_dim, n_actions))

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x


class DuelingDQN(nn.Module):
    def __init__(self, dims, n_actions, conv_net):
        super(DuelingDQN, self).__init__()
        self.conv_net = conv_net
        self.n_actions = n_actions
        self.layers = nn.ModuleList()
        prev_dim = dims[0]
        for dim in dims[1:]:
            self.layers.append(nn.Linear(prev_dim, dim))
            self.layers.append(nn.ReLU())
            prev_dim = dim
        self.layer_advantage = nn.Linear(prev_dim, n_actions)
        self.layer_value = nn.Linear(prev_dim, 1)

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        v = self.layer_value(x)
        a = self.layer_advantage(x)
        x = (
            v.expand(x.size(0), self.n_actions)
            + a
            - a.mean(1).unsqueeze(1).expand(x.size(0), self.n_actions)
        )
        return x


class ReccurentDDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layer, n_actions, conv_net):
        super(ReccurentDDQN, self).__init__()
        self.conv_net = conv_net
        self.n_actions = n_actions
        self.layers = nn.ModuleList()
        self.input_dim = input_dim

        self.layer_lstm = nn.LSTM(input_dim, hidden_dim, n_layer)
        self.layer_activation = nn.ReLU()
        self.layer_advantage = nn.Linear(hidden_dim, n_actions)
        self.layer_value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Convolutions
        conv_feats = torch.zeros(x.size(0), x.size(1), self.input_dim).to(x.device)
        for i in range(x.size(0)):
            conv_feats[i] = self.conv_net(x[i]).view(x[i].size(0), -1)
        # LSTM
        x, _ = self.layer_lstm(conv_feats)

        # Distributed FC
        out = []
        for i in range(x.size(0)):
            x[i] = self.layer_activation(x[i])
            v = self.layer_value(x[i])
            a = self.layer_advantage(x[i])
            out.append(
                v.expand(x[i].size(0), self.n_actions)
                + a
                - a.mean(1).unsqueeze(1).expand(x[i].size(0), self.n_actions)
            )
        return torch.stack(out)
