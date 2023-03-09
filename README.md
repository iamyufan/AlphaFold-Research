# AlphaFold-Research

Using AlphaFold and GNN to Predict Catalytic Efficiency of Enzymes

## Model Scratch

![model](img/scratch.png)

## Pipeline

### 0. Preparation

1. Create a [conda](https://docs.conda.io/en/latest/) environment with Python>3.7

```bash
conda create --name afgnn python=3.8
conda activate afgnn
```

2. Install required packages

```bash
pip install -r requirements.txt
```

### 1. Dimensionality Reduction

Use PCA to reduce the dimensionality of the `logits` in the output file `XXXXX.pkl` of modified AlphaFold.

`XXXXX.pkl` is a Python dict object denoting a AlphaFold output of a protein sequence with length of `e_dim`, whose `dict_keys` is given by

- 'distogram', 
  - 'bin_edges'
  - **'logits'**
- 'experimentally_resolved', 
- 'masked_msa', 
- 'predicted_lddt', 
- 'representations', 
  - 'msa'
  - 'msa_first_row'
  - 'pair'
  - **'single'**
  - 'structure_module'
- 'structure_module', 
- 'plddt'

We only need the **'logits'** and **'single'** for the enzyme features.

1. Modify the `./pca/pca.py` to change the `AF2_OUTPUT_DIR` to the corresponding path where the actual AlphaFold outputs are saved.

2. You may need to change the padding size if the maximum length of the `e_dim` is greater than 1600 in `./pca/pca.py`.

3. Conduct PCA on the pkl in `AF2_OUTPUT_DIR`

```bash
cd pca
nohup python3 pca.py >> log.txt
```

### 2. Feature Engineering

Construct the reaction graph as well as the node feature from the raw data.

1. Modify the `model_name` in the second block of the `./datasets/preprocess.ipynb` to match the target dataset (i.e. iMM904, iYO844 or iML1515)
   
2. Run through the code blocks in `./datasets/preprocess.ipynb`

3. [Optional] Modify the `train_test_split` ratio in the 19th code block to get different size of test set.

The output graph and feature will be saved to `./dataset/<dataset_name>/`.

- The node information with enzymes' single representation and molecules' feature are saved to `node.pkl` as a pandas dataframe.
- The link information is saved to `link.dat`.
- The training label is saved to `label.dat`.
- The test label is saved to `label.dat.test`.
- The logits feature of enzyme nodes are saved to `logits/X.npy` with `X` denotes the node index.

### 3. Network Training

1. Train the modified GCN

```bash
cd gnn
nohup python3 train.py --dataset iYO844 --num-layers 2 >> logs/iYO844.txt
```

## Acknowledgements

Code derived and reshaped from [HGB](https://github.com/THUDM/HGB).
