import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    MLP applies 2 layers of 2D/3D convolution with kernel_size=1
    and non-linearity to the channels of input
    
    Args:
    in_channels (int): input channels
    out_channels (int): output channels. Default is None. 
                        If None, same as in_channels
    hidden_channels (int): hidden channels. Default is None,
                            If None, same as in_channels
    n_dim (int) : 2D or 3D. Default is 2.
    non_linearity: Default is nn.functional.gelu
    dropout (float): Default is 0.0, If > 0, dropout probability
        
    """
    def __init__(self,
                 in_channels:int, 
                 out_channels=None, 
                 hidden_channels=None,
                 n_dim:int=2,
                 non_linearity=nn.functional.gelu,
                 dropout=0.0,
                 **kwargs
    ):
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.non_linearity = non_linearity
        n_layers = 2
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
            if dropout > 0.0
            else None
        )
        self.n_dim = n_dim
        
        if self.n_dim == 2:
            # input: [batchsize, in_channel=width, size_x+padding, size_y+padding]
            # weight: [hidden_channels=width, in_channel=width, 1,1]
            # output: [batchsize, out_channel=hidden_channels, size_x+padding, size_y+padding]
            # input: [batchsize, hidden_channels, size_x+padding, size_y+padding]
            # output: [batchsize, hidden_channels, size_x+padding, size_y+padding]
            # input: [batchsize, in_channel=hidden_channels=width, size_x+padding, size_y+padding]
            # weight: [out_channel=width, hidden_channels=width, 1, 1]
            # output: [batchsize, out_channel=width, size_x+padding, size_y+padding]
            self.mlp1 = nn.Conv2d(self.in_channels, self.hidden_channels, 1)
            self.mlp2 = nn.Conv2d(self.hidden_channels, self.out_channels, 1)
        else:
            # input: [batchsize, in_channel=width, size_x, size_y, T + padding]
            # weight: [hidden_channels=width, in_channel=width, 1,1,1]
            # output: [batchsize, out_channel=hidden_channels, size_x, size_y, T + padding]
            # input: [batchsize, hidden_channels, size_x, size_y, T + padding]
            # output: [batchsize, hidden_channels, size_x, size_y, T + padding]
            # input: [batchsize, in_channel=hidden_channels, size_x, size_y, T + padding]
            # weight: [out_channel=width, hidden_channels=width, 1, 1, 1]
            # output: [batchsize, out_channel=width, size_x, size_y, T + padding]
            self.mlp1 = nn.Conv3d(self.in_channels, self.hidden_channels, 1)
            self.mlp2 = nn.Conv3d(self.hidden_channels, self.out_channels, 1)
            

    def forward(self, x):
        x = self.mlp1(x)
        x = self.non_linearity(x)
        x = self.mlp2(x)
    
        return x
    
class ChannelMLP(nn.Module):
    """
    ChannelMLP applies an arbitrary number of layers of 
    1d convolution and nonlinearity to the channels of input
    and is invariant to spatial resolution.

    Args:
    in_channels (int): input channels
    out_channels (int): output channels. Default is None. 
                        If None, same as in_channels
    hidden_channels (int): hidden channels. Default is None,
                            If None, same as in_channels
    n_layers (int): number of layers. Default is 2.
    non_linearity: Default is nn.functional.gelu
    dropout (float): Default is 0.0, If > 0, dropout probability
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        hidden_channels=None,
        n_layers=2,
        non_linearity=nn.functional.gelu,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
            if dropout > 0.0
            else None
        )
        
        # we use nn.Conv1d for everything and roll data along the 1st data dim
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.in_channels, self.out_channels, 1))
            elif i == 0:
                self.fcs.append(nn.Conv1d(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.out_channels, 1))
            else:
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.hidden_channels, 1))

    def forward(self, x):
        reshaped = False
        size = list(x.shape)
        print(size)
        if x.ndim > 3:  
            # batch, channels, x1, x2... extra dims
            # .reshape() is preferable but .view()
            # cannot be called on non-contiguous tensors
            x = x.reshape((*size[:2], -1)) 
            reshaped = True
        print(x.shape)
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        # if x was an N-d tensor reshaped into 1d, undo the reshaping
        # same logic as above: .reshape() handles contiguous tensors as well
        if reshaped:
            x = x.reshape((size[0], self.out_channels, *size[2:]))

        return x


# Reimplementation of the ChannelMLP class using Linear instead of Conv
class LinearChannelMLP(torch.nn.Module):
    def __init__(self, layers, non_linearity=nn.functional.gelu, dropout=0.0):
        super().__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.fcs = nn.ModuleList()
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(self.n_layers)])
            if dropout > 0.0
            else None
        )

        for j in range(self.n_layers):
            self.fcs.append(nn.Linear(layers[j], layers[j + 1]))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        return x