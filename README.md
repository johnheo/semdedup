# Semantic Deduplication for Data Efficient Learning [[technical report]](https://drive.google.com/file/d/1z_ORLx81rJA8pgH5g4KucL5GJqVn3YNb/view?usp=sharing)
### 
> Jung Hwan Heo & Paul Chen

> Extension to [SemDeDup: Data-efficient learning at web-scale through semantic deduplication](https://arxiv.org/abs/2303.09540)

## Getting Started
To get the deduplicated dataloader and train a model, use the following command:
```py
python main.py \
    --ratio 25 \            # % data to use
    --n_clusters 20 \       # number of clusters
    --rank_type random \    # random vs. cossim
    --prune_type common \   # common vs. diverse
    --dataset fmnist        # mnist or fmnist
```
To generate a sweep over hyperparameters, simply edit the script `run.sh` and then launch 
```sh
sh run.sh
```
You can also interactively play around with our code through the two notebooks:
- `dedeup.ipynb` deduplicate with fast kmeans (w/ support on MPS backend)
- `train.ipynb` train with deduplicated dataset

## Introduction
- With the advent of big data, machine learning models are being trained on massive datasets but at diminishing returns. 
- These datasets usually have a considerable amount of redundant or duplicate data, which can result in longer training times, increased storage needs, and unnecessary computational complexity. 
- To overcome this issue, our project proposes utilizing clustering techniques for semantic deduplication, which can significantly reduce data redundancy and storage requirements while enhancing training efficiency.


## Objectives
- Dataset Exploration: Develop a semantic deduplication method using k-means clustering to identify and remove redundant data samples.
- Efficiency Tradeoff: Analyze the impact of semantic deduplication on storage requirements and training efficiency.

## Methods
Clustering techniques to be used
- [Fast K-means clustering](https://github.com/DeMoriarty/fast_pytorch_kmeans)
- [~~Gaussian Mixture Models~~](https://github.com/ldeecke/gmm-torch)

Models to be used for feature extraction
- VGG16
- ~~ResNet~~

Datasets used
- MNIST
- Fashion-MNIST
- CIFAR-10
- ~~Stanford Cars~~
- ~~Caltech101~~

Representation spaces to be used
- R1. Pixel space
- R2. Embedding space

Experiments
- E1. Pairwise Cosine similarity per clusters 
- E2. Data pruning ratio vs. Training steps vs. Performance

