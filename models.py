from torch import nn
import torch

class basic_conv_block(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(basic_conv_block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_ch, output_ch, 2, padding=1),
                                  nn.BatchNorm2d(output_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(output_ch, output_ch, 2, padding=1),
                                  nn.BatchNorm2d(output_ch),
                                  nn.ReLU(inplace=True))
    def forward(self, x):
        output = self.conv(x)
        return output
    

class lin_model(nn.Module):
    def __init__(self, n_channels, n_classes=3):
        super(lin_model, self).__init__()
        self.net = nn.Sequential(nn.Linear(n_channels, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 16),
                                 nn.ReLU(),
                                 nn.Linear(16, n_classes),
                                 nn.Softmax())
    def forward(self, x):
        x = torch.reshape(x, (-1,))
        out = self.net(x)
        return out
'''    
class conv_model(nn.Module):
    def __init__(self, n_channels, n_classes=3):
        super(conv_model, self).__init__()
        self.conv_block = nn.Sequential(nn.ConvTranspose2d(n_channels, 15, kernel_size=(2,2), stride=1),
                                 nn.BatchNorm2d(15),
                                 nn.ReLU(inplace=True),
#                                 nn.MaxPool2d(2),
                                 nn.ConvTranspose2d(15, 15, kernel_size=(2,2), stride=1),
                                 nn.BatchNorm2d(15),
                                 nn.ReLU(inplace=True),
#                                 nn.MaxPool2d(2),
                                 nn.ConvTranspose2d(15, 15, kernel_size=(2,2), stride=1),
                                 nn.BatchNorm2d(15),
                                 nn.ReLU(inplace=True),
#                                 nn.MaxPool2d(2),                                 
                                 nn.Conv2d(15, 5, kernel_size=(2,2), stride=1),
                                 nn.ReLU(inplace=True),
                                 nn.Flatten(),
                                 nn.Linear(200040, 256),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(256, 16),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(16, n_classes),
                                 nn.Softmax())
    def forward(self, x):
        out = self.conv_block(x)
        return out
'''    
class conv_model(nn.Module):
    def __init__(self, n_channels, n_classes=3):
        super(conv_model, self).__init__()
        self.conv_tr_block = nn.Sequential(nn.ConvTranspose2d(n_channels, 15, kernel_size=(2,2), stride=1),
                                 nn.BatchNorm2d(15),
                                 nn.ReLU(inplace=True),
#                                 nn.MaxPool2d(2),
                                 nn.ConvTranspose2d(15, 15, kernel_size=(2,2), stride=1),
                                 nn.BatchNorm2d(15),
                                 nn.ReLU(inplace=True),
#                                 nn.MaxPool2d(2),
                                 nn.ConvTranspose2d(15, 15, kernel_size=(2,2), stride=1),
                                 nn.BatchNorm2d(15),
                                 nn.ReLU(inplace=True))
#                                 nn.MaxPool2d(2),
        self.conv_block = nn.Sequential(nn.Conv2d(15, 5, kernel_size=(2,2), stride=1),
                                        nn.ReLU(inplace=True))
        self.flatten = nn.Flatten()
        self.linear_block = nn.Sequential(nn.Linear(200040, 256),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(256, 16),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(16, n_classes),
                                         nn.Softmax())
    def forward(self, x):
        x1 = self.conv_tr_block(x)
        x2 = self.conv_block(x1)
        x2 = self.flatten(x2)
        out = self.linear_block(x2)
        return out