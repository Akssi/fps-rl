
from CNN import *
from DQN import *
import numpy as np
from torchvision import transforms

# def test_CNN():
#     assert test_CNN_init()
#     assert test_CNN_forward()

def test_CNN_init():
    layers_params = [(3, 64, 3, 1, 0, 1)]
    model = CNN(layers_params, None)
    assert isinstance(model.layers[0], nn.Conv2d)
    assert isinstance(model.layers[1], nn.BatchNorm2d)
    assert isinstance(model.layers[2], nn.ReLU)
    assert model.layers[0].in_channels == 3
    assert model.layers[0].out_channels == 64
    assert model.layers[0].kernel_size == (3,3)
    assert model.layers[0].stride == (1,1)
    assert model.layers[0].padding == (0, 0)
    assert model.layers[0].dilation == (1, 1)

    return True

def test_CNN_forward():
    frame = torch.zeros((3, 108, 60))
    layers_params = [(3, 32, 8, 4, 0, 1)]
    model = CNN(layers_params, None)
    assert model(frame).shape == (1, 32, 26, 14)
    layers_params = [(3, 32, 8, 4, 0, 1), (32, 64, 4, 2, 0, 1)]
    model = CNN(layers_params, None)
    assert model(frame).shape == (1, 64, 12, 6)
    return True
    
def test_CNN_transform():
    frame = np.zeros((3, 108, 60), dtype=np.uint8)
    layers_params = [(1, 1, 1, 1, 0, 1)]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((54, 30)),
        transforms.ToTensor(),
        ])
    model = CNN(layers_params, transform)
    assert model(frame).shape == (1, 1, 54, 30)
    return True

def test_DQN_init():
    def dummy_cnn(x):
        return np.zeros((1, 64, 54, 30))
    dims = [54*30*64, 256, 512]
    n_actions = 10
    model = DQN(dims, n_actions, dummy_cnn)
    assert model.conv_net != None
    assert model.n_actions == 10
    assert isinstance(model.layers[0], nn.Linear)
    assert model.layers[0].in_features == 54*30*64
    assert model.layers[0].out_features == 256

def test_DQN_forward():
    def dummy_cnn(x):
        return torch.zeros((1, 64, 54, 30))
    dims = [54*30*64, 256, 512]
    n_actions = 10
    model = DQN(dims, n_actions, dummy_cnn)
    assert model(None).shape == (1, 10)

# def run_test():
#     test_CNN()

# if __name__ == "__main__":
#     run_test()