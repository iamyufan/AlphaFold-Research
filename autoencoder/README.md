# Autoencoder

The autoencoder is implemented to extract information from the output from Alphafold 2.

The data is formed as `BSUXXXXX.pkl`, which is a dictionary object. The representations are stored at `data['representations']`, which is a dictionary object whose keys include `['distogram', 'experimentally_resolved', 'masked_msa', 'predicted_lddt', 'representations', 'structure_module', 'plddt']`.

## Data Preprocess

The shape of the representations are not identical across BSUs. So, we need to preprocess the tensors to be the same shape.

### Method 1

**1. Pad the tensors to (2048, 2048, 128)**

Here, `zero_padding.py` is to take the tensors of the representations to be the shape of `(2048, 2048, 128)`, where the extra entries are filled with zeros.

**2. Treat each (2048, 2048, 128) tensor as one item**

### Method 2

**1. Pad the tensors to (2048, 2048, 128)**

Here, `zero_padding.py` is to take the tensors of the representations to be the shape of `(2048, 2048, 128)`, where the extra entries are filled with zeros.

**2. Split the tensor by channel**

Split the (2048, 2048, 128) tensor by channel so the one original tensor will be 128 tensors of size (2048, 2048, 1).

## Network Structure

### Conv2D Autoencoder (1)

The network architecture of the encoder is given by:

```python
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
    (1, 1, 1, 0),   # (16 * 16 * 1)
]
```

The network architecture of the decoder is given by:

```python
# (out_channels, kernel_size, stride, padding, output_padding)
decoder_architecture_config = [
    (1, 1, 1, 0, 0),   # (16 * 16 * 4)
    (4, 3, 1, 1, 0),   # (16 * 16 * 4)
    [(2, 1, 1, 0, 0), (4, 3, 1, 1, 0), 2],   # (16 * 16 * 4)
    (8, 3, 2, 1, 1),   # (32 * 32 * 8)
    [(4, 1, 1, 0, 0), (8, 3, 1, 1, 0), 2],   # (32 * 32 * 8)
    (16, 3, 2, 1, 1),   # (64 * 64 * 16)
    (32, 1, 1, 0, 0),  # (64 * 64 * 32)
    [(16, 1, 1, 0, 0), (32, 3, 1, 1, 0), 2],  # (64 * 64 * 32)
    (32, 3, 2, 1, 1),  # (128 * 128 * 32)
    (64, 1, 1, 0, 0),  # (128 * 128 * 64)
    [(32, 1, 1, 0, 0), (64, 3, 1, 1, 0), 2],  # (128 * 128 * 64)
    (64, 3, 2, 1, 1),  # (256 * 256 * 64)
    (96, 3, 2, 1, 1),  # (512 * 512 * 96)
    (128, 5, 2, 2, 1),  # (1024 * 1024 * 128)
    (128, 5, 1, 2, 0),  # (1024 * 1024 * 128)
    (128, 5, 2, 2, 1),  # (2048 * 2048 * 128)
    (128, 3, 1, 1, 0),  # (2048 * 2048 * 128)
]
```

### Conv2D Autoencoder (2)

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
```
