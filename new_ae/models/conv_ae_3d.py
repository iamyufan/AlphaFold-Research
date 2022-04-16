import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision


# (out_channels, kernel_size, stride, padding)
encoder_architecture_config = [
    # input: 2048x2048x128
    (1, (4, 4, 4), 4, 1),
    # 512x512x32
    (1, (4, 4, 4), 4, 1),
    # 128x128x8
    (1, (4, 4, 4), 4, 1),
    # 32x32x2
]

# (out_channels, kernel_size, scale)
decoder_architecture_config = [
    # encoded: 32x32x2
    (1, (1, 1, 1), 4),
    # 128x128x8
    (1, (1, 1, 1), 4),
    # 512x512x32
    (1, (1, 1, 1), 4),
    # 2048x2048x128
]

class Conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, k_size, stride=1, p=1, num_groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Sequential( 
            nn.Conv3d(ch_in, ch_out, kernel_size=k_size, stride=stride, padding=p),  
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        out = self.conv(x)
        return out


class up_conv(nn.Module):
    "Reduce the number of features by 2 using Conv with kernel size 1x1x1 and double the spatial dimension using 3D trilinear upsampling"
    def __init__(self, ch_in, ch_out, k_size=1, scale=2, align_corners=False):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=k_size),
            nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=align_corners),
        )
    def forward(self, x):
        return self.up(x)
    
    
class Conv3DAutoEncoder(nn.Module):
    def __init__(self, in_channel=128, **kwargs):
        super(Conv3DAutoEncoder, self).__init__()
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
                    Conv_block(
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
                        Conv_block(
                            in_channels,
                            conv1[0],
                            kernel_size=conv1[1],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        Conv_block(
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
                    up_conv(
                        in_channels, 
                        x[0], 
                        kernel_size=x[1], 
                        scale=x[2]
                    )
                ]
                in_channels = x[0]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        up_conv(
                            in_channels,
                            conv1[0],
                            kernel_size=conv1[1],
                            scale=conv1[2]
                        )
                    ]
                    layers += [
                        up_conv(
                            conv1[0],
                            conv2[0],
                            kernel_size=conv2[1],
                            scale=conv2[2]
                        )
                    ]
                    in_channels = conv2[0]

        layers += Conv_block(1, 1, k_size=3, stride=1, p=1)
        return nn.Sequential(*layers)