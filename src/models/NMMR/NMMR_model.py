import torch
import torch.nn as nn


class MLP_for_NMMR(nn.Module):
    
    def __init__(self, input_dim, train_params):
        super(MLP_for_NMMR, self).__init__()

        self.train_params = train_params
        self.network_width = train_params["network_width"]
        self.network_depth = train_params["network_depth"]

        self.layer_list = nn.ModuleList()
        for i in range(self.network_depth):
            if i == 0:
                self.layer_list.append(nn.Linear(input_dim, self.network_width))
            else:
                self.layer_list.append(nn.Linear(self.network_width, self.network_width))
        self.layer_list.append(nn.Linear(self.network_width, 1))

    def forward(self, x):
        for ix, layer in enumerate(self.layer_list):
            if ix == (self.network_depth + 1):  # if last layer, don't apply relu activation
                x = layer(x)
            else:
                x = torch.relu(layer(x))

        return x


class cnn_for_dsprite(nn.Module):
    def __init__(self, train_params):
        super(cnn_for_dsprite, self).__init__()
        self.train_params = train_params
        self.batch_size = train_params["batch_size"]

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # A blocks
        self.conv_A_0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding='same')
        self.conv_A_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.conv_A_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.conv_A_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.projection_A = nn.Linear(128*16*16, 128)

        # W blocks
        self.conv_W_0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding='same')
        self.conv_W_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.conv_W_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.conv_W_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.projection_W = nn.Linear(128*16*16, 128)

        # MLP
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, A, W):
        # A head
        A = torch.relu(self.conv_A_0(A))
        A = torch.relu(self.conv_A_1(A))
        A = self.max_pool(A)
        A = torch.relu(self.conv_A_2(A))
        A = torch.relu(self.conv_A_3(A))
        A = self.max_pool(A)
        A = torch.flatten(A, start_dim=1)
        A = self.projection_A(A)

        # W head
        W = torch.relu(self.conv_W_0(W))
        W = torch.relu(self.conv_W_1(W))
        W = self.max_pool(W)
        W = torch.relu(self.conv_W_2(W))
        W = torch.relu(self.conv_W_3(W))
        W = self.max_pool(W)
        W = torch.flatten(W, start_dim=1)
        W = self.projection_W(W)

        x = torch.cat((A, W), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x
