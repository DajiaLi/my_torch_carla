import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary


class DQN_model_64x3_CNN(nn.Module):
    """
    input:[b, 300, 400]
    """
    def __init__(self):
        super(DQN_model_64x3_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.fc1 = nn.Linear(11520, 3)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.avg_pool2d(out, kernel_size=5, stride=3, padding=(2, 2))  # torch.Size([2, 64, 100, 134])
        out = F.relu(self.conv2(out))
        out = F.avg_pool2d(out, kernel_size=5, stride=3, padding=(2, 2))  # torch.Size([2, 64, 34, 45])
        out = F.relu(self.conv3(out))
        out = F.avg_pool2d(out, kernel_size=5, stride=3, padding=(2, 2))  # torch.Size([2, 64, 12, 15])
        out = out.view(out.size(0), -1)  # torch.Size([2, 11520])
        out = self.fc1(out)
        return out

class DQN_model_64x3_CNN_abstract(nn.Module):
    """
    input:[b, 600, 400]
    """
    def __init__(self):
        super(DQN_model_64x3_CNN_abstract, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.fc1 = nn.Linear(22080, 3)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.avg_pool2d(out, kernel_size=5, stride=3, padding=(2, 2))  # torch.Size([2, 64, 100, 134])
        # print(out.shape)
        out = F.relu(self.conv2(out))
        out = F.avg_pool2d(out, kernel_size=5, stride=3, padding=(2, 2))  # torch.Size([2, 64, 34, 45])
        # print(out.shape)
        out = F.relu(self.conv3(out))
        out = F.avg_pool2d(out, kernel_size=5, stride=3, padding=(2, 2))  # torch.Size([2, 64, 12, 15])
        # print(out.shape)
        out = out.view(out.size(0), -1)  # torch.Size([2, 22080])
        # print(out.shape)
        out = self.fc1(out)
        return out

class DDPG_action2_model_64x3_CNN_abstract(nn.Module):
    """
    input:[b, 600, 400]
    """
    def __init__(self):
        super(DDPG_action2_model_64x3_CNN_abstract, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.fc1 = nn.Linear(22080, 2)

    def forward(self, x, get_feature=False):
        out = F.relu(self.conv1(x))
        out = F.avg_pool2d(out, kernel_size=5, stride=3, padding=(2, 2))  # torch.Size([2, 64, 100, 134])
        out = F.relu(self.conv2(out))
        out = F.avg_pool2d(out, kernel_size=5, stride=3, padding=(2, 2))  # torch.Size([2, 64, 34, 45])
        out = F.relu(self.conv3(out))
        out = F.avg_pool2d(out, kernel_size=5, stride=3, padding=(2, 2))  # torch.Size([2, 64, 12, 15])
        out = out.view(out.size(0), -1)  # torch.Size([2, 22080])
        if get_feature:
            return out
        out = self.fc1(out)
        return out

class DDPG_value2_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = torch.nn.Sequential(
            nn.Linear(22080, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.last_liner = nn.Linear(128 + 2, 1)
    def forward(self, state, action):
        state = self.sequential(state)
        input = torch.cat([state, action], dim=1)
        x = self.last_liner(input)
        return x

if __name__ == '__main__':
    net = DQN_model_64x3_CNN()
    device = torch.device('cuda')
    input = torch.rand([10, 3, 300, 400])
    print(net(input).shape)
    net = DQN_model_64x3_CNN_abstract()
    device = torch.device('cuda')
    input = torch.rand([10, 3, 600, 400])
    print(net(input).shape)
    # net.to(device)
    # summary(net, (3, 600, 400))
