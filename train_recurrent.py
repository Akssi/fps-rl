from vizdoom import *
import random
import math
import time
from models import DQN, CNN
from collections import namedtuple
from torchvision import transforms
import torch.nn.functional as F
import torch
import numpy as np
import wandb

random.seed(69)
torch.manual_seed(83)
np.random.seed(87)

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity, empty_transition):
        self.capacity = capacity
        self.empty_transition = empty_transition
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, seq_len):
        index_sample = random.sample(list(range(len(self.memory))), batch_size)
        sample = []
        for index in index_sample:
            seq = [self.empty_transition for _ in range(seq_len)]
            slice_end = min(index + seq_len, len(self.memory))
            diff = slice_end - index
            seq[:diff] = self.memory[index:slice_end]
            mask = torch.zeros(len(seq))
            for i, t in enumerate(seq):
                if t == 0 or t.next_state is None:
                    mask[: i + 1] = 1
                    break
            sample.append(np.array(seq))
        return np.array(sample), mask

    def __len__(self):
        return len(self.memory)


def select_action(img, steps):
    eps_treshold = wandb.config.eps_end + (
        wandb.config.eps_start - wandb.config.eps_end
    ) * math.exp(-1.0 * steps / wandb.config.eps_decay)
    if random.random() > eps_treshold:
        return state_action_values.argmax(1)
    else:
        return torch.tensor(random.randint(0, n_actions - 1)).unsqueeze(0)


def optimize_model():
    if len(memory) < wandb.config.batch_size:
        return 0

    print("optim")
    transitions, mask_batch = memory.sample(
        wandb.config.batch_size, wandb.config.seq_len
    )
    transitions = transitions.swapaxes(0, 1)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # non_final_mask = torch.tensor(
    #     tuple(map(lambda s: s is not None, transitions[:, :, 2])),
    #     device=device,
    #     dtype=torch.bool,
    # )
    # non_final_next_states = torch.cat(
    #     [s for s in transitions[:, :, 2] if s is not None]
    # )
    print([transitions[0, b, 2] is not None for b in range(wandb.config.batch_size)])
    state_batch = torch.stack(
        [
            torch.cat([transitions[s_i, b, 1] for b in range(wandb.config.batch_size)])
            for s_i in range(wandb.config.seq_len)
        ]
    )
    # action_batch = torch.stack(
    #     [
    #         torch.cat([transitions[s_i, b, 1] for b in range(wandb.config.batch_size)])
    #         for s_i in range(wandb.config.seq_len)
    #     ],
    #     device=device,
    #     dtype=torch.Long,
    # )
    # non_final_mask = torch.tensor(
    #     [
    #         torch.tensor(
    #             [
    #                 transitions[s_i, b, 2] is not None
    #                 for b in range(wandb.config.batch_size)
    #             ],
    #             device=device,
    #             dtype=torch.bool,
    #         )
    #         for s_i in range(wandb.config.seq_len)
    #     ]
    # )
    # non_final_next_states = torch.stack(
    #     [
    #         torch.cat(
    #             [
    #                 transitions[s_i, b, 2]
    #                 for b in range(wandb.config.batch_size)
    #                 if transitions[s_i, b, 2] is not None
    #             ]
    #         )
    #         for s_i in range(wandb.config.seq_len)
    #     ]
    # )
    # reward_batch = torch.stack(
    #     [
    #         torch.cat([transitions[s_i, b, 3] for b in range(wandb.config.batch_size)])
    #         for s_i in range(wandb.config.seq_len)
    #     ]
    # )
    print(non_final_mask.shape)
    print(non_final_next_states.shape)
    quit()

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch.view(-1, 1))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states).max(1)[0].detach()
    )
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    state_action_values *= mask
    expected_state_action_values *= mask
    # Compute Huber loss
    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss


def push_to_tensor(tensor, x):
    return torch.cat((tensor[1:], x.unsqueeze(0)))


# DQN Settings
BATCH_SIZE = 3
SEQ_LEN = 2
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 15
LR = 1e-3
default_config = {
    "model": "RecurrentDDQN-Replay",
    "lr": LR,
    "batch_size": BATCH_SIZE,
    "seq_len": SEQ_LEN,
    "n_hidden": 128,
    "n_lstm_layer": 1,
    "img_shape": (45, 30),
    "target_update": TARGET_UPDATE,
    "eps_start": EPS_START,
    "eps_end": EPS_END,
    "eps_decay": EPS_DECAY,
    "episodes": 10000,
}

wandb.init(project="fps-rl", config=default_config)

game = DoomGame()
game.load_config("scenarios/basic.cfg")
game.init()

shoot = [0, 0, 1]
left = [1, 0, 0]
right = [0, 1, 0]
actions = [shoot, left, right]
n_actions = len(actions)


if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
else:
    device = "cpu"

empty_transition = Transition(
    torch.zeros(1, 1, wandb.config.img_shape[0], wandb.config.img_shape[1]).to(device),
    torch.zeros(n_actions).to(device),
    torch.zeros(1, wandb.config.img_shape[0], wandb.config.img_shape[1]).to(device),
    0,
)
# Define Convnet
ConvLayer = namedtuple(
    "ConvLayer",
    ["in_channel", "out_channel", "kernel", "stride", "padding", "dilation"],
)
conv_layers = [ConvLayer(1, 8, 6, 3, 0, 1), ConvLayer(8, 8, 3, 2, 0, 1)]
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize(wandb.config.img_shape),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0).to(device)),
    ]
)
conv_net = CNN.CNN(conv_layers, None).to(device)

# Define DQN nets
policy_net = DQN.ReccurentDDQN(
    4 * 6 * 8, wandb.config.n_hidden, wandb.config.n_lstm_layer, n_actions, conv_net
).to(device)
target_net = DQN.ReccurentDDQN(
    4 * 6 * 8, wandb.config.n_hidden, wandb.config.n_lstm_layer, n_actions, conv_net
).to(device)
target_net.eval()
wandb.watch(policy_net)
wandb.watch(target_net)

optimizer = torch.optim.Adam(policy_net.parameters(), wandb.config.lr)
memory = ReplayMemory(10000, empty_transition)

steps = 0
for i in range(wandb.config.episodes):
    game.new_episode()
    episode_actions = np.array([0 for _ in range(n_actions)])

    # Initial state from env
    state = game.get_state()
    # Tensor hold episode running sequence of states (shape seq x batch x channel x width x height)
    img_seq = torch.zeros(
        wandb.config.seq_len,
        1,
        1,
        wandb.config.img_shape[0],
        wandb.config.img_shape[1],
    ).to(device)
    while not game.is_episode_finished():
        # Unpack current state
        img = transform(state.screen_buffer)
        img_seq = push_to_tensor(img_seq, img)
        # Compute Q(s,a)
        state_action_values = policy_net(img_seq)[-1]

        # Choose action
        action = select_action(img_seq, steps).to(device)

        # Take action
        one_hot_action = F.one_hot(action, n_actions).squeeze().tolist()
        episode_actions += np.array(one_hot_action)
        reward = game.make_action(one_hot_action)
        reward = torch.tensor([reward], device=device)

        if not game.is_episode_finished():
            next_state = game.get_state()
            next_img = transform(next_state.screen_buffer)
        else:
            next_state = None
            next_img = None

        # Store the transition in memory
        memory.push(img, action, next_img, reward)

        # Update state and model
        state = next_state
        loss = optimize_model()
        steps += 1

    if i % wandb.config.target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
    logs = {
        "episode_reward": game.get_total_reward(),
        "loss": loss,
        "episode_actions_count": episode_actions.sum(),
    }
    for i in range(n_actions):
        logs[f"episode_actions_{i}"] = episode_actions[i] / episode_actions.sum()
    wandb.log(logs)

