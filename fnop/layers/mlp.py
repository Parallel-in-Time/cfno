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
                 n_dim=2,
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
        
        if n_dim == 2:
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
