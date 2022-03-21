import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision

# (out_channels, kernel_size, stride, padding)
encoder_architecture_config = [
    (128, 7, 4, 3),
    (64, 5, 2, 2),
    (64, 3, 2, 1),  # (128 * 128 * 64)
    [(32, 1, 1, 0), (64, 3, 1, 1), 2],  # (128 * 128 * 64)
    (32, 1, 1, 0),  # (128 * 128 * 32)
    (32, 3, 2, 1),  # (64 * 64 * 32)
    [(16, 1, 1, 0), (32, 3, 1, 1), 2],  # (64 * 64 * 32)
    (16, 1, 1, 0),  # (64 * 64 * 16)
    (8, 3, 2, 1),   # (32 * 32 * 8)
    [(4, 1, 1, 0), (8, 3, 1, 1), 2],    # (32 * 32 * 8)
    (4, 3, 2, 1),   # (16 * 16 * 4)
    [(2, 1, 1, 0), (4, 3, 1, 1), 2],    # (16 * 16 * 4)
    (4, 3, 1, 1),   # (16 * 16 * 4)
    (1, 1, 1, 0)    # (16 * 16 * 1)
]

# (out_channels, kernel_size, stride, padding)
decoder_architecture_config = [
    (1, 1, 1, 0)    # (16 * 16 * 4)
    (4, 3, 1, 1),   # (16 * 16 * 4)
    [(2, 1, 1, 0), (4, 3, 1, 1), 2],   # (16 * 16 * 4) 
    (4, 3, 2, 1),   # (32 * 32 * 8)
    [(4, 1, 1, 0), (8, 3, 1, 1), 2],   # (32 * 32 * 8)
    (8, 3, 2, 1),   # (64 * 64 * 16)
    (16, 1, 1, 0),  # (64 * 64 * 32)
    [(16, 1, 1, 0), (32, 3, 1, 1), 2],  # (64 * 64 * 32)
    (32, 3, 2, 1),  # (128 * 128 * 32)
    (32, 1, 1, 0),  # (128 * 128 * 64)
    [(32, 1, 1, 0), (64, 3, 1, 1), 2],  # (128 * 128 * 64)
    (64, 3, 2, 1),  # (256 * 256 * 64)
    (64, 5, 2, 2),  # (512 * 512 * 128)
    (128, 7, 4, 3), # (2048 * 2048 * 128)
]

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class ConvTranspose2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvTranspose2dBlock, self).__init__()
        self.convtrans = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.convtrans(x)))
    
    
class Conv2DAutoEncoder(nn.Module):
    def __init__(self, in_channel=128, **kwargs):
        super(Conv2DAutoEncoder, self).__init__()
        self.encoder_archi = encoder_architecture_config
        self.decoder_archi = decoder_architecture_config
        self.encoder, self.latent_channels = self._build_encoder(self.encoder_archi, in_channel)
        self.decoder = self._build_decoder(self.decoder_archi, self.latent_channels)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def _build_encoder(self, architecture, in_channel):
        layers = []
        in_channels = in_channel
        for x in architecture:
            if type(x) == tuple:
                layers += [
                    Conv2dBlock(
                        in_channels, x[0], kernel_size=x[1], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[0]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        Conv2dBlock(
                            in_channels,
                            conv1[0],
                            kernel_size=conv1[1],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        Conv2dBlock(
                            conv1[0],
                            conv2[0],
                            kernel_size=conv2[1],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[0]

        return nn.Sequential(*layers), in_channels

    def _build_decoder(self, architecture, latent_channels):
        layers = []
        in_channels = latent_channels
        for x in architecture:
            if type(x) == tuple:
                layers += [
                    ConvTranspose2dBlock(
                        in_channels, x[0], kernel_size=x[1], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[0]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        ConvTranspose2dBlock(
                            in_channels,
                            conv1[0],
                            kernel_size=conv1[1],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        ConvTranspose2dBlock(
                            conv1[0],
                            conv2[0],
                            kernel_size=conv2[1],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[0]

        return nn.Sequential(*layers)