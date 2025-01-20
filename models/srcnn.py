from torch import nn

class SRCNN(nn.Module):
    def __init__(self, do_padding = True, in_channels = 3):
        super().__init__()
        self.do_padding = do_padding
        self.in_channels = in_channels
        
        if do_padding: padding_var = 1
        else: padding_var = 0

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=9, padding=9 // 2 * padding_var, padding_mode='replicate')
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=self.in_channels, kernel_size=5, padding=5 // 2 * padding_var, padding_mode='replicate')
        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.relu(self.conv1(X))
        X = self.relu(self.conv2(X))
        y = self.conv3(X)

        return y