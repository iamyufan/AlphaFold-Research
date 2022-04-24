# Autoencoder

The autoencoder is implemented to extract information from the output from AlphaFold 2.

The output tensor data from AlphaFold is stored in pickle file, which is a dictionary object. The representations are stored at `data['representations']`, which is a dictionary object whose keys include `['distogram', 'experimentally_resolved', 'masked_msa', 'predicted_lddt', 'representations', 'structure_module', 'plddt']`.

## Data Preprocess

The shape of the representations are not identical across enzymes. So, we need to preprocess the tensors to be the same shape.

### Zero Padding

**1. Pad the tensors to (2048, 2048, 128)**

Here, `zero_padding.py` is to take the tensors of the representations to be the shape of `(2048, 2048, 128)`, where the extra entries are filled with zeros.

**2. Treat each (2048, 2048, 128) tensor as one item**

<!-- ### Method 2

**1. Pad the tensors to (2048, 2048, 128)**

Here, `zero_padding.py` is to take the tensors of the representations to be the shape of `(2048, 2048, 128)`, where the extra entries are filled with zeros.

**2. Split the tensor by channel**

Split the (2048, 2048, 128) tensor by channel so the one original tensor will be 128 tensors of size (2048, 2048, 1). -->

## Network Structure

### Conv3D Autoencoder

The network architecture of the encoder is given by:

```python
# (out_channels, kernel_size, stride, padding)
encoder_architecture_config = [
    # input: 2048x2048x128
    (1, (4, 4, 4), 4, 1),
    # 512x512x32
    (1, (4, 4, 4), 4, 1),
    # 128x128x8
    (1, (4, 4, 4), 4, 1),
    # 32x32x2
    (1, (2, 2, 2), 2, 0),
    # 16x16x1
]
```

The network architecture of the decoder is given by:

```python
# (out_channels, kernel_size, scale)
decoder_architecture_config = [
    # encoded: 16x16x1
    (1, (1, 1, 1), 2),
    # 32x32x2
    (1, (1, 1, 1), 4),
    # 128x128x8
    (1, (1, 1, 1), 4),
    # 512x512x32
    (1, (1, 1, 1), 4),
    # 2048x2048x128
]
```

The overall architecture of the autoencoder is given by:

```python
Conv3DAutoEncoder(
  (encoder): Sequential(
    (0): Conv_block(
      (conv): Sequential(
        (0): Conv3d(1, 1, kernel_size=(4, 4, 4), stride=(4, 4, 4), padding=(1, 1, 1))
        (1): BatchNorm3d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
    )
    (1): Conv_block(
      (conv): Sequential(
        (0): Conv3d(1, 1, kernel_size=(4, 4, 4), stride=(4, 4, 4), padding=(1, 1, 1))
        (1): BatchNorm3d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
    )
    (2): Conv_block(
      (conv): Sequential(
        (0): Conv3d(1, 1, kernel_size=(4, 4, 4), stride=(4, 4, 4), padding=(1, 1, 1))
        (1): BatchNorm3d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
    )
    (3): Conv_block(
      (conv): Sequential(
        (0): Conv3d(1, 1, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        (1): BatchNorm3d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
    )
  )
  (decoder): Sequential(
    (0): up_conv(
      (up): Sequential(
        (0): Conv3d(1, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): Upsample(scale_factor=2.0, mode=trilinear)
      )
    )
    (1): up_conv(
      (up): Sequential(
        (0): Conv3d(1, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): Upsample(scale_factor=4.0, mode=trilinear)
      )
    )
    (2): up_conv(
      (up): Sequential(
        (0): Conv3d(1, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): Upsample(scale_factor=4.0, mode=trilinear)
      )
    )
    (3): up_conv(
      (up): Sequential(
        (0): Conv3d(1, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        (1): Upsample(scale_factor=4.0, mode=trilinear)
      )
    )
    (4): Conv_block(
      (conv): Sequential(
        (0): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (1): BatchNorm3d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
    )
  )
)
```

<!-- ### Conv2D Autoencoder (2)

The network architecture of the encoder is given by:

```python
# (out_channels, kernel_size, stride, padding)
encoder_architecture_config = [
    (128, 7, 4, 3),
    (64, 5, 2, 2),
    (32, 3, 2, 1),  # (128 * 128 * 64)
    (16, 3, 2, 1),  # (64 * 64 * 32)
    (8, 3, 2, 1),   # (32 * 32 * 8)
    (4, 3, 2, 1),   # (16 * 16 * 4)
    (1, 1, 1, 0),   # (16 * 16 * 1)
]
```

The network architecture of the decoder is given by:

```python
# (out_channels, kernel_size, stride, padding, output_padding)
decoder_architecture_config = [
    (4, 3, 1, 1, 0),   # (16 * 16 * 4)
    (8, 3, 2, 1, 1),   # (32 * 32 * 8)
    (16, 3, 2, 1, 1),   # (64 * 64 * 16)
    (32, 1, 1, 0, 0),  # (64 * 64 * 32)
    (64, 3, 2, 1, 1),  # (128 * 128 * 64)
    (64, 3, 2, 1, 1),  # (256 * 256 * 64)
    (96, 3, 2, 1, 1),  # (512 * 512 * 96)
    (128, 5, 2, 2, 1),  # (1024 * 1024 * 128)
    (128, 5, 1, 2, 0),  # (1024 * 1024 * 128)
    (128, 3, 1, 1, 0),  # (2048 * 2048 * 128)
]
``` -->
