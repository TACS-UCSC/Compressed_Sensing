import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    @staticmethod
    def compl_mul2d(input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights) ## batch, input, lat, lon EIN TENSOR PRODUCT input, output, lat, lon

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes...works, don't use the one below
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        out_ft[..., :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[..., :self.modes1, :self.modes2], self.weights1)
        out_ft[..., -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[..., -self.modes1:, :self.modes2], self.weights2)

        # out_ft1 = self.compl_mul2d(x_ft[:,:, :self.modes1, :self.modes2], self.weights1)
        # out_ft2 = self.compl_mul2d(x_ft[:,:, -self.modes1:, :self.modes2], self.weights2)

        # out_ft1 = torch.nn.functional.pad(out_ft1,(0,self.modes1,0,self.modes2))
        # out_ft2 = torch.nn.functional.pad(out_ft2,(self.modes1,0,0,self.modes2))

        # out_ft = out_ft1 + out_ft2
        
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

class FNO2D_grid(nn.Module):
    def __init__(self, 
                 modes1 = 20, 
                 modes2 = 20, 
                 width = 10, 
                 padding = 8, 
                 channels = 3, 
                 channelsout = 3, 
                 gridx=[0,1], 
                 gridy=[0,1], 
                 padval = 0, 
                 pbias = True,
                 fourier_layers = 4
                 ):
        super(FNO2D_grid, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.gridx = gridx
        self.gridy = gridy
        self.padding = padding
        self.padval = padval # pad the domain if input is non-periodic
        self.p = nn.Linear(channels+2, self.width, bias = pbias) 

        self.spectral_layers = nn.ModuleList([])

        for layer in range(fourier_layers):
            self.spectral_layers.append(nn.ModuleDict({
                                                        "sconv" : SpectralConv2d(self.width, self.width, self.modes1, self.modes2),
                                                        "mlp" : MLP(self.width, self.width, self.width), ## 2 layer linear, gelu activation
                                                        "w" : nn.Conv2d(self.width, self.width, kernel_size=1, bias=True), ## use Conv2d so you dont have to permute the tensors dimensions
                                                     })
                                       )

        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, channelsout, self.width)
        
    def forward(self, x):
        # Add grid coordinates to the end of the input array for FNO prediction x, y coordinates
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        
        # Lift the input to the desired channel dimension
        x = self.p(x)
        
        # Permute the tensor dimensions to match expected input format for convolutional layers
        x = x.permute(0, 3, 1, 2)
        
        # Apply padding if specified
        if self.padding is not None:
            x = F.pad(x, [self.padding, self.padding, self.padding, self.padding], mode="constant", value=self.padval)
        
        # Apply spectral convolution layers
        for spectral_layer in self.spectral_layers:
            x1 = spectral_layer["mlp"](self.norm(spectral_layer["sconv"](self.norm(x))))
            x2 = spectral_layer["w"](x)  # Linear transform
            x = F.gelu(x1 + x2)

        # Project from the channel space to the output space
        x = self.q(x)
        
        # Permute the tensor dimensions back to original format
        x = x.permute(0, 2, 3, 1)
        
        # Remove padding if specified
        if self.padding is not None:
            x = x[:, self.padding:-self.padding, self.padding:-self.padding, :]

        return x

    def get_grid(self, shape, device):
        ## modifications to include lat/lon coordinates being inputted into the function
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(self.gridx, dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(self.gridy, dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class FNO2D_grid_tembedding(nn.Module):
    def __init__(self, 
                 modes1 = 20, 
                 modes2 = 20, 
                 width = 10, 
                 padding = None, 
                 in_channels = 3, 
                 out_channels = 3, 
                 gridx=None, 
                 gridy=None, 
                 padval = 0, 
                 pbias = True,
                 fourier_layers = 4
                 ):
        super(FNO2D_grid_tembedding, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.gridx = gridx
        self.gridy = gridy
        self.padding = padding
        self.padval = padval # pad the domain if input is non-periodic
        # self.p = nn.Linear(channels+2, self.width, bias = pbias) 
        self.p = MLP(in_channels+2, self.width, self.width)

        self.spectral_layers = nn.ModuleList([])

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(width),
                nn.Linear(width, width),
                nn.ReLU()
            )

        for layer in range(fourier_layers):
            self.spectral_layers.append(nn.ModuleDict({
                                                        "sconv" : SpectralConv2d(self.width, self.width, self.modes1, self.modes2),
                                                        "mlp" : MLP(self.width, self.width, self.width), ## 2 layer linear, gelu activation
                                                        "w" : nn.Conv2d(self.width, self.width, kernel_size=1, bias=True), ## use Conv2d so you dont have to permute the tensors dimensions
                                                     })
                                       )

        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, out_channels, self.width)
        
    def forward(self, x, timestep):
        # Add grid coordinates to the end of the input array for FNO prediction x, y coordinates
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        
        # Lift the input to the desired channel dimension
        x = self.p(x)
        
        # Permute the tensor dimensions to match expected input format for convolutional layers

        # Apply padding if specified
        if self.padding is not None:
            x = F.pad(x, [self.padding, self.padding, self.padding, self.padding], mode="constant", value=self.padval)
        
        # Add time embedding
        t_emb = self.time_mlp(timestep)
        t_emb = t_emb[(...,) + (None,) * 2]  # Reshape to (batch, width, 1, 1)
        x = x + t_emb ## sufficient to add the embedding, due to high dimensionality of the input
        
        # Apply spectral convolution layers
        for spectral_layer in self.spectral_layers:
            x1 = spectral_layer["mlp"](self.norm(spectral_layer["sconv"](self.norm(x))))
            x2 = spectral_layer["w"](x)  # Linear transform
            x = F.gelu(x1 + x2)
            # x = x + t_emb

        # Project from the channel space to the output space
        x = self.q(x)
                
        # Remove padding if specified
        if self.padding is not None:
            x = x[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return x

    def get_grid(self, shape, device):
        ## modifications to include lat/lon coordinates being inputted into the function
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(self.gridx, dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x,  1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(self.gridy, dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

class FNO2D_grid_tembedding_cond(nn.Module):
    def __init__(self, 
                 modes1 = 20, 
                 modes2 = 20, 
                 width = 10, 
                 padding = None, 
                 in_channels = 3, 
                 out_channels = 3, 
                 gridx=None, 
                 gridy=None, 
                 padval = 0, 
                 pbias = True,
                 fourier_layers = 4
                 ):
        super(FNO2D_grid_tembedding_cond, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.gridx = gridx
        self.gridy = gridy
        self.padding = padding
        self.padval = padval # pad the domain if input is non-periodic
        # self.p = nn.Linear(channels+2, self.width, bias = pbias) 
        self.p = MLP(in_channels+2, self.width, self.width)

        self.spectral_layers = nn.ModuleList([])

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(width),
                nn.Linear(width, width),
                nn.ReLU()
            )

        for layer in range(fourier_layers):
            self.spectral_layers.append(nn.ModuleDict({
                                                        "sconv" : SpectralConv2d(self.width, self.width, self.modes1, self.modes2),
                                                        "mlp" : MLP(self.width, self.width, self.width), ## 2 layer linear, gelu activation
                                                        "w" : nn.Conv2d(self.width, self.width, kernel_size=1, bias=True), ## use Conv2d so you dont have to permute the tensors dimensions
                                                     })
                                       )

        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, out_channels, self.width)
        
    def forward(self, x, cond, timestep):
        # Add grid coordinates to the end of the input array for FNO prediction x, y coordinates
        ## adding conditional to input
        x = x + cond

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        # Lift the input to the desired channel dimension
        x = self.p(x)
        
        # Permute the tensor dimensions to match expected input format for convolutional layers

        # Apply padding if specified
        if self.padding is not None:
            x = F.pad(x, [self.padding, self.padding, self.padding, self.padding], mode="constant", value=self.padval)
        
        # Add time embedding
        t_emb = self.time_mlp(timestep)
        t_emb = t_emb[(...,) + (None,) * 2]  # Reshape to (batch, width, 1, 1)
        x = x + t_emb ## sufficient to add the embedding, due to high dimensionality of the input
        
        # Apply spectral convolution layers
        for spectral_layer in self.spectral_layers:
            x1 = spectral_layer["mlp"](self.norm(spectral_layer["sconv"](self.norm(x))))
            x2 = spectral_layer["w"](x)  # Linear transform
            x = F.gelu(x1 + x2)
            # x = x + t_emb

        # Project from the channel space to the output space
        x = self.q(x)
                
        # Remove padding if specified
        if self.padding is not None:
            x = x[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return x

    def get_grid(self, shape, device):
        ## modifications to include lat/lon coordinates being inputted into the function
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(self.gridx, dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x,  1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(self.gridy, dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)
