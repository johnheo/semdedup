# Semantic Deduplication for Data Efficient Learning

> Extension to [SemDeDup: Data-efficient learning at web-scale through semantic deduplication](https://arxiv.org/abs/2303.09540)


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
- [Gaussian Mixture Models](https://github.com/ldeecke/gmm-torch)

Representation spaces to be used
- R1. Pixel space
- R2. Embedding space

Models to be used for feature extraction
- ResNet
- VGG
- ViT
- CLIP 

Datasets to be used
- MNIST
- Fashion MNIST
- CIFAR-10,100
- Stanford Cars
- Caltech101

Experiments
- E1. Pairwise Cosine similarity per clusters 
- E2. Data pruning ratio vs. Training steps vs. Performance

## Roadmap
- [ ] E1 on R1 and R2 for MNIST datasets
- [ ] Train and Evaluate with Lightweight CNN

